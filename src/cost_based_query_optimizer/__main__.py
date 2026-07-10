"""CLI de demostración: `cost-based-query-optimizer {demo,benchmark}`.

Este optimizador no incluye un parser SQL (por diseño — ver `../AGENTS.md`,
sin imports cruzados con `sql-query-parser`), así que su CLI no puede
aceptar texto SQL como las de sus hermanos. En su lugar, `demo` construye a
mano un puñado de `SelectStatement`/`Catalog` ilustrativos —directamente
con los dataclasses de `models.py`, exactamente la forma en que `nanosql`
los construiría a partir de un AST real— y muestra el plan físico elegido
en formato `EXPLAIN`; `benchmark` mide el tiempo de planificación según
crece el número de tablas de una cadena de `INNER JOIN`, para enseñar dónde
está realmente el punto de cruce entre la reordenación exhaustiva (DP) y la
heurística greedy (`physical.MAX_DP_TABLES`)."""

from __future__ import annotations

import argparse
import time
from collections.abc import Sequence

from .models import (
    BinaryOp,
    BinaryOperator,
    Catalog,
    ColumnRef,
    ColumnStatistics,
    FunctionCall,
    IndexStatistics,
    Join,
    JoinType,
    Literal,
    LiteralType,
    SelectItem,
    SelectStatement,
    Star,
    TableRef,
    TableStatistics,
)
from .physical import MAX_DP_TABLES
from .pipeline import explain, optimize


def _index_selection_catalog() -> Catalog:
    orders = TableStatistics(
        row_count=100_000,
        columns={
            "id": ColumnStatistics(distinct_count=100_000),
            "customer_id": ColumnStatistics(distinct_count=100_000),
            "status": ColumnStatistics(distinct_count=3),
        },
        indexes=(
            IndexStatistics(name="idx_orders_customer_id", columns=("customer_id",)),
            IndexStatistics(name="idx_orders_status", columns=("status",)),
        ),
    )
    return Catalog(tables={"orders": orders})


def _demo_index_vs_full_scan() -> None:
    print("=== 1. Selección de índice vs. full scan (misma tabla, dos predicados) ===\n")
    catalog = _index_selection_catalog()

    selective = SelectStatement(
        columns=(SelectItem(Star()),),
        from_table=TableRef("orders"),
        where=BinaryOp(
            ColumnRef("customer_id"), BinaryOperator.EQ, Literal(42, LiteralType.INTEGER)
        ),
    )
    print("-- WHERE customer_id = 42  (alta selectividad: 1 / 100000 filas) --")
    print(explain(optimize(selective, catalog).physical_plan))
    print()

    unselective = SelectStatement(
        columns=(SelectItem(Star()),),
        from_table=TableRef("orders"),
        where=BinaryOp(
            ColumnRef("status"), BinaryOperator.EQ, Literal("PENDING", LiteralType.STRING)
        ),
    )
    print("-- WHERE status = 'PENDING'  (baja selectividad: ~1/3 de las filas) --")
    print(explain(optimize(unselective, catalog).physical_plan))
    print()


def _join_reorder_catalog() -> Catalog:
    customers = TableStatistics(row_count=100, columns={"id": ColumnStatistics(distinct_count=100)})
    orders = TableStatistics(
        row_count=100_000,
        columns={
            "id": ColumnStatistics(distinct_count=100_000),
            "customer_id": ColumnStatistics(distinct_count=100),
        },
    )
    order_items = TableStatistics(
        row_count=500_000,
        columns={
            "id": ColumnStatistics(distinct_count=500_000),
            "order_id": ColumnStatistics(distinct_count=100_000),
        },
    )
    return Catalog(tables={"customers": customers, "orders": orders, "order_items": order_items})


def _demo_join_reordering() -> None:
    print("=== 2. Reordenación de joins (3 tablas, escritas en el peor orden) ===\n")
    catalog = _join_reorder_catalog()
    select = SelectStatement(
        columns=(SelectItem(Star()),),
        from_table=TableRef("order_items", alias="oi"),
        joins=(
            Join(
                JoinType.INNER,
                TableRef("orders", alias="o"),
                BinaryOp(ColumnRef("order_id", "oi"), BinaryOperator.EQ, ColumnRef("id", "o")),
            ),
            Join(
                JoinType.INNER,
                TableRef("customers", alias="c"),
                BinaryOp(ColumnRef("customer_id", "o"), BinaryOperator.EQ, ColumnRef("id", "c")),
            ),
        ),
    )
    print("-- Escrito como: order_items (500k filas) JOIN orders (100k) JOIN customers (100) --")
    print(explain(optimize(select, catalog).physical_plan))
    print()


def _demo_aggregation() -> None:
    print("=== 3. Agregación (GROUP BY + COUNT) ===\n")
    catalog = _join_reorder_catalog()
    select = SelectStatement(
        columns=(
            SelectItem(ColumnRef("customer_id", "o"), alias="customer_id"),
            SelectItem(FunctionCall(name="COUNT", star_argument=True), alias="total"),
        ),
        from_table=TableRef("orders", alias="o"),
        group_by=(ColumnRef("customer_id", "o"),),
    )
    print("-- SELECT customer_id, COUNT(*) FROM orders GROUP BY customer_id --")
    print(explain(optimize(select, catalog).physical_plan))
    print()


def _demo_outer_join_scope() -> None:
    print("=== 4. LEFT JOIN: orden preservado (alcance documentado, ver README) ===\n")
    catalog = _join_reorder_catalog()
    select = SelectStatement(
        columns=(SelectItem(Star()),),
        from_table=TableRef("order_items", alias="oi"),
        joins=(
            Join(
                JoinType.LEFT,
                TableRef("orders", alias="o"),
                BinaryOp(ColumnRef("order_id", "oi"), BinaryOperator.EQ, ColumnRef("id", "o")),
            ),
        ),
    )
    print(
        "-- order_items (500k) LEFT JOIN orders (100k): no se reordena aunque orders sea"
        " más pequeño --"
    )
    print(explain(optimize(select, catalog).physical_plan))
    print()


def _cmd_demo(_args: argparse.Namespace) -> int:
    _demo_index_vs_full_scan()
    _demo_join_reordering()
    _demo_aggregation()
    _demo_outer_join_scope()
    return 0


def _benchmark_catalog(table_count: int) -> Catalog:
    return Catalog(
        tables={
            f"t{i}": TableStatistics(
                row_count=1_000, columns={"id": ColumnStatistics(distinct_count=1_000)}
            )
            for i in range(table_count)
        }
    )


def _benchmark_select(table_count: int) -> SelectStatement:
    joins = tuple(
        Join(
            JoinType.INNER,
            TableRef(f"t{i}", alias=f"t{i}"),
            BinaryOp(ColumnRef("id", f"t{i - 1}"), BinaryOperator.EQ, ColumnRef("id", f"t{i}")),
        )
        for i in range(1, table_count)
    )
    return SelectStatement(
        columns=(SelectItem(Star()),), from_table=TableRef("t0", alias="t0"), joins=joins
    )


def _cmd_benchmark(_args: argparse.Namespace) -> int:
    print("Tiempo de planificación según crece una cadena de INNER JOIN (todas contra 'id').")
    print(f"Umbral DP -> greedy: MAX_DP_TABLES = {MAX_DP_TABLES}\n")
    print(f"{'tablas':>7} | {'motor':>7} | {'tiempo (ms)':>12}")
    print("-" * 32)
    for table_count in range(2, 17):
        catalog = _benchmark_catalog(table_count)
        select = _benchmark_select(table_count)
        engine = "DP" if table_count <= MAX_DP_TABLES else "greedy"
        start = time.perf_counter()
        optimize(select, catalog)
        elapsed_ms = (time.perf_counter() - start) * 1000
        print(f"{table_count:>7} | {engine:>7} | {elapsed_ms:>12.2f}")
    return 0


def _build_arg_parser() -> argparse.ArgumentParser:
    arg_parser = argparse.ArgumentParser(
        prog="cost-based-query-optimizer",
        description="Cost-based query optimizer: AST -> logical plan -> physical plan.",
    )
    subparsers = arg_parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("demo", help="run illustrative optimization scenarios")
    subparsers.add_parser(
        "benchmark", help="measure planning time as the number of joined tables grows"
    )
    return arg_parser


def main(argv: Sequence[str] | None = None) -> int:
    arg_parser = _build_arg_parser()
    args = arg_parser.parse_args(argv)
    handlers = {"demo": _cmd_demo, "benchmark": _cmd_benchmark}
    return handlers[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
