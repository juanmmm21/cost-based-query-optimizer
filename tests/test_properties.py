"""Tests de propiedades: genera catálogos y consultas de join aleatorios con
semilla fija y verifica que el coste que elige la reordenación de joins
(`physical._dp_join_order`, vía `optimize()`) es siempre igual al mínimo
encontrado por una búsqueda exhaustiva de fuerza bruta sobre las `n!`
ordenaciones left-deep posibles — un cálculo de referencia independiente en
cuanto a ESTRATEGIA DE BÚSQUEDA (enumeración explícita de permutaciones
frente a programación dinámica sobre subconjuntos), que es exactamente la
parte de `physical.py` con más riesgo de bug (un subconjunto mal cubierto,
un desempate incorrecto, un predicado aplicado en el paso equivocado). Las
fórmulas de coste/cardinalidad en sí ya están cubiertas de forma aislada en
`test_cost_model.py`/`test_cardinality.py`, así que la fuerza bruta
reutiliza deliberadamente las mismas piezas de bajo nivel de `physical.py`
(`_plan_scan`, `_choose_physical_join`, `_applicable_at`, `_build_pool`) en
vez de reimplementarlas — lo que se contrasta es la exploración, no el
modelo de coste (ver docstring de `test_join_order.py` para el mismo
razonamiento aplicado a casos concretos en vez de aleatorios)."""

import itertools
import math
import random

import pytest
from support import cmp, eq_columns, inner_join, ref, star_select

from cost_based_query_optimizer import physical as _physical
from cost_based_query_optimizer.models import (
    BinaryOperator,
    Catalog,
    ColumnStatistics,
    Join,
    JoinType,
    SelectStatement,
    TableRef,
    TableStatistics,
)
from cost_based_query_optimizer.pipeline import optimize
from cost_based_query_optimizer.protocols import SimpleCardinalityEstimator, SimpleCostModel

COST_MODEL = SimpleCostModel()
ESTIMATOR = SimpleCardinalityEstimator()

#: `n!` de fuerza bruta: 7! = 5040 ordenaciones por escenario, suficiente
#: para estresar la reordenación sin que el test tarde segundos.
MAX_BRUTE_FORCE_TABLES = 7
SCENARIOS_PER_SEED = 12
SEEDS = (1, 2, 3, 4, 5)


def _random_scenario(rng: random.Random) -> tuple[list[TableRef], Catalog, SelectStatement]:
    n = rng.randint(3, MAX_BRUTE_FORCE_TABLES)
    names = [f"t{i}" for i in range(n)]
    refs = [TableRef(name, name) for name in names]

    tables_stats: dict[str, TableStatistics] = {}
    for name in names:
        row_count = rng.randint(20, 5000)
        columns = {
            "id": ColumnStatistics(distinct_count=row_count),
            "val": ColumnStatistics(distinct_count=max(1, rng.randint(1, row_count))),
        }
        tables_stats[name] = TableStatistics(row_count=row_count, columns=columns)
    catalog = Catalog(tables=tables_stats)

    # Árbol de expansión: cada tabla (salvo la primera) se une por igualdad
    # de "id" con una tabla anterior aleatoria — garantiza que el grafo de
    # joins sea conexo, sin restringir el orden en que la reordenación debe
    # explorarlo.
    joins: list[Join] = []
    for i in range(1, n):
        parent = rng.randrange(i)
        joins.append(
            inner_join(refs[i], eq_columns(ref("id", refs[i].alias), ref("id", refs[parent].alias)))
        )

    # Predicados extra de una sola tabla (selectividad sobre "val") y
    # aristas redundantes de dos tablas en el WHERE (además del árbol de
    # arriba), para ejercitar el pushdown y el "conjunto aplicable en más
    # de un candidato" simultáneamente.
    where_conjuncts = []
    for i in range(n):
        if rng.random() < 0.5:
            where_conjuncts.append(
                cmp(ref("val", refs[i].alias), BinaryOperator.GT, rng.randint(0, 100))
            )
    extra_edges = rng.randint(0, max(0, n - 2))
    for _ in range(extra_edges):
        i, j = rng.sample(range(n), 2)
        where_conjuncts.append(eq_columns(ref("id", refs[i].alias), ref("id", refs[j].alias)))

    where = None
    if where_conjuncts:
        where = where_conjuncts[0]
        for extra in where_conjuncts[1:]:
            from cost_based_query_optimizer.models import BinaryOp

            where = BinaryOp(left=where, operator=BinaryOperator.AND, right=extra)

    select = star_select(refs[0], joins=tuple(joins), where=where)
    return refs, catalog, select


def _pool(select: SelectStatement) -> list:
    pool = list(_physical.split_conjuncts(select.where))
    for join in select.joins:
        pool.extend(_physical.split_conjuncts(join.on))
    return pool


def _brute_force_minimum_cost(
    tables: list[TableRef], select: SelectStatement, catalog: Catalog
) -> float:
    conjuncts = _physical._build_pool(_pool(select), tables, catalog)
    n = len(tables)
    base = [
        _physical._plan_scan(
            tables[i],
            _physical._applicable_at(conjuncts, 0, i),
            catalog,
            COST_MODEL,
            ESTIMATOR,
        )
        for i in range(n)
    ]

    best = math.inf
    for permutation in itertools.permutations(range(n)):
        subset = 1 << permutation[0]
        current = base[permutation[0]]
        for t in permutation[1:]:
            applicable = _physical._applicable_at(conjuncts, subset, t)
            current = _physical._choose_physical_join(
                current,
                _physical._tables_in_mask(tables, subset),
                base[t],
                tables[t],
                _physical.combine_conjuncts(applicable) if applicable else None,
                JoinType.INNER,
                catalog,
                COST_MODEL,
                ESTIMATOR,
            )
            subset |= 1 << t
        best = min(best, current.estimated_cost)
    return best


@pytest.mark.parametrize("seed", SEEDS)
def test_dp_join_order_always_matches_brute_force_minimum(seed: int) -> None:
    rng = random.Random(seed)
    for _ in range(SCENARIOS_PER_SEED):
        tables, catalog, select = _random_scenario(rng)
        optimized_cost = optimize(select, catalog).estimated_cost
        brute_force_cost = _brute_force_minimum_cost(tables, select, catalog)
        assert optimized_cost == pytest.approx(brute_force_cost, rel=1e-9, abs=1e-9)


@pytest.mark.parametrize("seed", SEEDS)
def test_dp_join_order_is_deterministic_across_repeated_runs(seed: int) -> None:
    rng = random.Random(seed)
    for _ in range(SCENARIOS_PER_SEED):
        _, catalog, select = _random_scenario(rng)
        first = optimize(select, catalog).physical_plan
        second = optimize(select, catalog).physical_plan
        assert first == second
