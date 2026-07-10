# cost-based-query-optimizer

Part of [`strata-database-engine`](https://github.com/juanmmm21/strata-database-engine), a from-scratch relational database engine built as a set of independent, individually-usable components. This repository is the query engine's optimizer: [`juanmmm21/cost-based-query-optimizer`](https://github.com/juanmmm21/cost-based-query-optimizer).

## What it is and what problem it solves

A cost-based query optimizer turns a SQL `SELECT` into a logical plan (a tree of relational operators — scan, filter, join, aggregate, sort) and then into a physical plan (a tree of concrete algorithms — sequential scan or index scan, nested-loop or hash join) chosen by estimating and comparing the cost of the alternatives, not by fixed heuristics.

Concretely, this module:

- Estimates predicate selectivity from table statistics — row counts, per-column distinct-value counts, and optional equi-height histograms for range predicates.
- Decides, for every scan, whether a full table scan or an available secondary index is cheaper, given the actual selectivity of the query's predicates against that specific table.
- Reorders the joins of a multi-table query (bounded dynamic programming over subsets, falling back to a greedy heuristic for large joins) to minimize total estimated cost, instead of executing joins in the order they were written.
- Chooses, for each join, between a nested-loop and a hash join based on estimated cost.
- Does all of this **deterministically**: the same AST against the same statistics always produces the same plan, with no unseeded randomness anywhere in the search.

## Role in `strata-database-engine`

```text
sql-query-parser ──▶ cost-based-query-optimizer ──▶ vectorized-execution-engine
                              ▲
                              │  table statistics
                              │
              bplus-tree-storage-engine / lsm-tree-engine
```

`sql-query-parser` produces a typed AST for a `SELECT` statement. This module consumes an AST-shaped input (see "Data format" below) plus a `Catalog` of table statistics — cardinality, per-column distinct counts, histograms, available indexes — that in the assembled system would come from whichever storage engine is active (`bplus-tree-storage-engine` or `lsm-tree-engine`). It produces a physical plan that `vectorized-execution-engine` would execute. As with every pair of sibling repositories in this ecosystem, there is no direct import between them — each repository defines its own equivalent types, and the real wiring happens only inside `nanosql`, the assembled engine.

## Goal / skills demonstrated

- **Cost-based search over a combinatorial space**: join order has factorial complexity in the number of tables; this module implements the classic Selinger/System R dynamic-programming approach (restricted to left-deep plans, with a documented greedy fallback above a configurable table-count threshold) rather than a brute-force or purely heuristic search.
- **Statistical estimation**: selectivity for equality, range (via histograms), `AND`/`OR`/`NOT` combinations (independence assumption), and the classic containment formula for join cardinality — every constant and simplifying assumption named and justified in the source, never a bare magic number.
- **Correctness under combinatorial search**: getting a cost-based join reordering search *provably* optimal under its own cost model is subtler than it looks — see the "Benchmarks" section below for a real bug this project's own property-based tests caught before publishing.
- **Deterministic design under complexity**: the join-reordering search never depends on `dict`/`set`/hash iteration order — only on bitmasks over an explicitly sorted table list — which is what makes "same input → same plan, always" actually provable rather than merely usually true.
- **Documented scope boundaries**: outer joins are not freely reorderable without changing query semantics, so this optimizer explicitly declines to reorder or push predicates across a `LEFT`/`RIGHT`/`FULL JOIN`, falling back to written-order planning with cost-based method selection only — a real, common simplification in production optimizers, not swept under the rug.

## How it works

### Pipeline

```text
SelectStatement ──▶ build_logical_plan ──▶ LogicalOperator tree
                                                    │
                                                    ▼
                                          build_physical_plan
                                                    │
                                                    ▼
                                          PhysicalOperator tree
                                          (EXPLAIN-style, with
                                           estimated_rows/cost
                                           on every node)
```

`logical.py` is a direct structural translation of the AST — no rewriting happens there. All cost-guided decisions (predicate pushdown, index selection, join method, join order) happen in a single pass in `physical.py`.

### The predicate-applicability mechanism

Every `WHERE` conjunct and every `JOIN ... ON` condition is split on `AND` into a flat pool. Each conjunct is applied at exactly one point in plan construction: the step that introduces the *last* table it references. Applied to a single scan (the base case), this is predicate pushdown and index-column matching; applied to a join step, it is attaching the right `ON`/`WHERE` condition at the earliest point the reordering search reaches it. One mechanism, not two — see `physical.py`'s module docstring for the short proof that this covers every conjunct exactly once, for any insertion order.

### Join reordering: left-deep DP with a greedy fallback

For a chain of `n` tables joined with `INNER JOIN`, `_dp_join_order` computes, for every subset of tables, the cheapest way to reach it by adding one table at a time to the cheapest plan for the rest of the subset (`O(n · 2ⁿ)`, restricted to left-deep trees — the same restriction many real Selinger-style optimizers make to keep the search tractable). Above `MAX_DP_TABLES` (12), a greedy heuristic takes over: start from the cheapest single table, then repeatedly add whichever remaining table yields the lowest incremental cost.

Making the DP's "keep only the cheapest sub-plan per subset" rule *correct* requires that a subset's estimated row count never depend on which insertion order was used to reach it — otherwise a cheaper sub-plan discarded for its higher cost could still have been the better base for further joins. This module's `CardinalityEstimator.estimate_join_cardinality` is deliberately designed so every recognized join predicate contributes its own independent, order-invariant reduction factor (built from *raw*, unfiltered catalog statistics rather than a table's cardinality after prior filtering) — see the "Benchmarks" section for the real bug this fixed.

### Scan selection

For every table, the optimizer computes the cost of a full scan against the cost of every available secondary index whose leading column matches part of the applicable predicate, and picks whichever is cheaper — an index only wins when the matched fraction of the table is actually small enough that random per-row access beats a sequential scan.

### Outer joins: a documented scope boundary

A `LEFT`/`RIGHT`/`FULL JOIN` cannot be freely reordered or have `WHERE` pushed across it without changing the query's result (a predicate on the nullable side can silently turn an outer join into an inner one). If any join in a query's `FROM` chain is not `INNER`, this optimizer preserves the written order for the whole chain and applies the full `WHERE` as a single residual filter at the end — it still chooses the cheapest physical method (nested-loop vs. hash) per join, just not the order.

## Architecture

```text
src/cost_based_query_optimizer/
├── models.py       # Input AST subset (mirrors sql-query-parser's shape, no
│                     import) + TableStatistics/ColumnStatistics/Catalog
├── errors.py       # Typed exception hierarchy (OptimizerError and subclasses)
├── predicates.py   # Conjunct split/combine, ColumnRef -> table resolution
├── protocols.py    # CostModel / CardinalityEstimator Protocols + defaults
├── plan.py         # LogicalOperator and PhysicalOperator trees
├── logical.py      # AST -> logical plan (structural, no rewriting)
├── physical.py     # logical plan -> physical plan: scan/join selection,
│                     left-deep DP + greedy join reordering
├── pipeline.py      # optimize() entry point + EXPLAIN-style plan printer
└── __main__.py     # CLI: demo, benchmark
```

`CostModel` and `CardinalityEstimator` are the only two pluggable `Protocol`s — the same pattern as `ErrorFormatter` in `sql-query-parser` or `DeadlockVictimPolicy` in `lock-manager-deadlock-detector`. `nanosql` (or any other consumer) can inject a cost model calibrated to real hardware, or a cardinality estimator backed by richer statistics, without touching `physical.py`.

## Requirements and installation

- Python `>=3.11`
- No runtime dependencies.

```bash
git clone https://github.com/juanmmm21/cost-based-query-optimizer.git
cd cost-based-query-optimizer
pip install -e ".[dev]"
```

## Usage

### CLI

```bash
cost-based-query-optimizer demo        # four illustrative scenarios with EXPLAIN output
cost-based-query-optimizer benchmark   # planning-time scaling, DP vs. greedy
```

`demo` builds a handful of `SelectStatement`/`Catalog` objects directly with this module's own dataclasses (there is no SQL parser here — see "Data format" below) and prints the chosen physical plan for: index vs. full scan selection, join reordering across a three-table query written in the worst order, a `GROUP BY`/`COUNT(*)` aggregate, and a `LEFT JOIN` demonstrating the outer-join scope boundary.

### Programmatic usage

```python
from cost_based_query_optimizer.models import (
    BinaryOp, BinaryOperator, Catalog, ColumnRef, ColumnStatistics,
    IndexStatistics, Literal, LiteralType, SelectItem, SelectStatement,
    Star, TableRef, TableStatistics,
)
from cost_based_query_optimizer.pipeline import explain, optimize

catalog = Catalog(tables={
    "orders": TableStatistics(
        row_count=100_000,
        columns={
            "id": ColumnStatistics(distinct_count=100_000),
            "customer_id": ColumnStatistics(distinct_count=100_000),
        },
        indexes=(IndexStatistics(name="idx_customer_id", columns=("customer_id",)),),
    ),
})

query = SelectStatement(
    columns=(SelectItem(Star()),),
    from_table=TableRef("orders"),
    where=BinaryOp(
        left=ColumnRef("customer_id"),
        operator=BinaryOperator.EQ,
        right=Literal(42, LiteralType.INTEGER),
    ),
)

result = optimize(query, catalog)
print(explain(result.physical_plan))
print(result.estimated_cost, result.estimated_rows)
```

## Data format / interface exposed to `nanosql`

Per `../AGENTS.md`, there is no cross-repo import between this module and `sql-query-parser`: `models.py` **redefines**, independently, the subset of the AST shape a cost-based optimizer needs to consume — `SelectStatement` and everything reachable from it (`Expression` and its variants, `SelectItem`, `TableRef`, `Join`, `OrderByItem`). Field names and dataclass shapes deliberately match `sql-query-parser`'s AST one-for-one, so `nanosql` can construct this module's input directly from that module's parsed output without any adapter layer — the coupling is "the two shapes agree," wired together for real only inside `nanosql`. DML/DDL statement shapes are out of scope here: a cost-based optimizer plans reads, not `INSERT`/`UPDATE`/`DELETE`/`CREATE`.

The other half of the input contract is `models.Catalog` — the table statistics a storage engine would supply: `TableStatistics` (row count, per-column `ColumnStatistics` with distinct-value counts and optional equi-height `HistogramBucket` sequences for range selectivity, and available `IndexStatistics`).

The output is `plan.PhysicalOperator` — the tree `vectorized-execution-engine` would execute, where every node carries its own `estimated_rows`/`estimated_cost` (the root's `estimated_cost` is the total estimated cost of the whole query, the number the optimizer minimized to choose that plan).

## Development

```bash
pip install -e ".[dev]"
pytest
ruff check .
ruff format --check .
mypy --strict src/
```

85 tests across ten files:

- `test_predicates.py` (18) / `test_cardinality.py` (17) / `test_cost_model.py` (12): conjunct split/combine, `ColumnRef` → table resolution, every selectivity/cardinality/cost formula in isolation (including the symmetric, order-invariant containment formula for join cardinality).
- `test_scan_selection.py` (7): index chosen when selective, full scan chosen when not, residual filters kept when a predicate only partially matches an index column — the DoD's "index vs. full scan by cost, not fixed heuristics."
- `test_join_order.py` (4): known 3- and 4-table scenarios, including an exhaustive check that the optimizer's cost is never beaten by *any* of the `3!` manually-assembled left-deep alternatives for a fixed join graph — the DoD's "optimal join order for 3+ table queries."
- `test_aggregation_sort_limit.py` (7) / `test_outer_join_scope.py` (3) / `test_errors.py` (4): `GROUP BY`/`ORDER BY`/`LIMIT` plan shape, the documented outer-join scoping decision, and the typed exception hierarchy.
- `test_determinism.py` (3): repeated `optimize()` calls and catalogs built with reversed `dict` insertion order produce byte-for-byte identical plans — CLAUDE.md's determinism rule (same AST + same statistics → same plan, always), checked as a black-box property rather than assumed from the implementation.
- `test_properties.py` (10): a seeded (`random.Random(seed)`, 5 fixed seeds) generator of random join graphs (spanning trees plus redundant extra edges, up to 7 tables) and statistics, each checked with an *independent* verification strategy — not a reimplementation of the cost formulas (those are already covered above), but an exhaustive brute-force search over all `n!` left-deep orderings using the same low-level building blocks, verifying the DP's chosen cost always equals the true minimum.

## Benchmarks

```text
Planning time as a chain of INNER JOINs grows (all on 'id'), MAX_DP_TABLES = 12:

 tables |  engine |  time (ms)
--------------------------------
      2 |      DP |       0.07
      6 |      DP |       0.77
      9 |      DP |       9.58
     12 |      DP |     107.68
     13 |  greedy |       0.25
     16 |  greedy |       0.34
```

The exhaustive DP roughly doubles in planning time per extra table, as expected from its `O(n · 2ⁿ)` complexity — at 12 tables it already costs over 100ms just to *plan* the query. The greedy fallback stays flat regardless of table count, at the price of no longer being guaranteed globally optimal. `MAX_DP_TABLES` is the explicit, documented threshold where this project chooses to cross over (`physical.py`); run `cost-based-query-optimizer benchmark` to reproduce.

A real finding, not a hypothetical one: this project's own property-based tests (`test_properties.py`) initially caught the DP search returning a *more expensive* plan than an exhaustive brute-force search could find for the same query — a genuine correctness bug, not just a performance concern. The root cause was that the join-cardinality estimate for a given subset of tables could come out differently depending on which insertion order was used to reach it (a duplicated join condition got double-counted, a single-table predicate got applied twice, and the containment formula only looked at one side of each join asymmetrically) — which breaks the optimal-substructure assumption the entire DP recurrence depends on. Fixed by making every recognized join predicate contribute its own fixed, order-invariant reduction factor built from raw catalog statistics; see the `fix(physical)` commit and `protocols.py`'s `estimate_join_cardinality` docstring for the full account.

## Troubleshooting

- **`MissingStatisticsError`**: the query references a table absent from the `Catalog` passed to `optimize()`. This optimizer never assumes a default row count for a missing table — silently guessing would produce a plan that looks fine but is baseless.
- **`UnresolvedColumnError`**: an unqualified `ColumnRef` (no `table=`) matched zero or more than one table's known columns in the `Catalog`. Qualify the column (`t.column`) or ensure only one joined table's `TableStatistics.columns` contains that name.
- **`UnsupportedQueryError`**: either the input `SelectStatement`'s join chain isn't left-deep (not producible by `logical.build_logical_plan`, only reachable by constructing a `LogicalJoin` tree by hand), or another construction outside this module's documented scope was fed directly to `physical.build_physical_plan`.
- **A `LEFT`/`RIGHT`/`FULL JOIN` query's plan looks unreordered / an obviously-smaller table isn't joined first**: expected — see "Outer joins: a documented scope boundary" above. Only chains where every join is `INNER` get reordered.
- **Planning a large join is slow**: check the table count against `physical.MAX_DP_TABLES` (12). Above it, planning already uses the greedy fallback; below it, the exponential DP is inherent to exhaustive left-deep search (see "Benchmarks") — lower `MAX_DP_TABLES` if a project needs faster planning at the cost of exhaustive search on medium-sized joins.

## License

MIT — see [`LICENSE`](LICENSE).
