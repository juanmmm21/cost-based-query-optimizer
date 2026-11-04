"""Construcción del plan físico a partir del plan lógico: selección de
índice vs. full scan, elección de método de join (nested-loop vs. hash) y
reordenación de joins — todo guiado por `CostModel`/`CardinalityEstimator`
(`protocols.py`), de forma determinista.

## Mecanismo unificado de aplicación de predicados

`logical.build_logical_plan` nunca mueve un predicado de sitio: el `WHERE`
queda como un único `LogicalFilter` por encima de toda la cadena de joins, y
cada `ON` se queda en su `LogicalJoin`. Este módulo junta TODOS esos
conjuntos (los de cada `ON`, más los del `WHERE`, partidos por `AND` con
`predicates.split_conjuncts`) en un único pool, y decide en qué paso de la
construcción del plan se aplica cada uno con una única regla: un conjunto
`c` se aplica exactamente en el paso que añade la ÚLTIMA tabla que
referencia — es decir, cuando el conjunto de tablas ya acumuladas más la
tabla que se añade en ese paso cubre exactamente las tablas que toca `c`.

Esa regla, aplicada al caso base (un único scan, "acumulado" vacío + la
propia tabla), es exactamente el pushdown de predicados de una sola tabla
hacia su scan (incluida la selección de índice); aplicada a cada paso de
join, es exactamente adjuntar cada condición de `ON`/`WHERE` en el punto
más temprano posible de la reordenación. Es la misma idea que "aplicar un
predicado tan pronto como todas sus columnas estén disponibles" de
cualquier optimizador de libro de texto, expresada como un único mecanismo
en vez de dos (pushdown de scan y adjunción de join) tratados por separado.
Se demuestra formalmente en el docstring de `_applicable_at` que, para
cualquier orden de inserción de tablas, cada conjunto del pool se aplica
en EXACTAMENTE un paso (nunca cero, nunca dos).

## Reordenación de joins: DP left-deep, con fallback greedy

`_dp_join_order` es una variante restringida a árboles **left-deep** del
clásico algoritmo de programación dinámica de Selinger/System R: para cada
subconjunto de tablas, el mejor plan es el mejor, entre todas las tablas
`t` del subconjunto, de "el mejor plan del resto del subconjunto, unido con
`t`". Restringir a left-deep (en vez de explorar también formas "bushy") es
una simplificación real y documentada frente a un optimizador de
producción: mantiene la complejidad en `O(n · 2ⁿ)` en vez de peor, y es
exactamente lo que hacen muchos optimizadores basados en Selinger en la
práctica. Por encima de `MAX_DP_TABLES` tablas se usa `_greedy_join_order`
(construcción incremental: en cada paso añade la tabla que produce el coste
total más bajo) — mismo principio, sin explorar las `n!` ordenaciones.

## Alcance: sólo se reordenan cadenas 100% `INNER JOIN`

Un `LEFT`/`RIGHT`/`FULL OUTER JOIN` no es libremente reordenable ni
separable de su `WHERE` sin cambiar el resultado (un predicado de `WHERE`
sobre el lado que puede tener NULLs generados por el propio outer join
convierte el join, de facto, en uno interno — un error clásico de
optimizadores ingenuos). Por eso, si la cadena de joins de una consulta
contiene algún join no-`INNER`, este módulo renuncia a reordenar y a
empujar predicados: conserva el orden escrito, sólo decide método físico
por coste en cada paso, y aplica el `WHERE` completo como un único filtro
final. Ver `_plan_join_chain_no_reorder`.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

from .errors import UnsupportedQueryError
from .models import (
    BinaryOp,
    BinaryOperator,
    Catalog,
    ColumnRef,
    Expression,
    FunctionCall,
    JoinType,
    Literal,
    TableRef,
    TableStatistics,
)
from .plan import (
    LogicalAggregate,
    LogicalDistinct,
    LogicalFilter,
    LogicalJoin,
    LogicalLimit,
    LogicalOperator,
    LogicalProject,
    LogicalScan,
    LogicalSort,
    PhysicalDistinct,
    PhysicalFilter,
    PhysicalHashAggregate,
    PhysicalHashJoin,
    PhysicalIndexScan,
    PhysicalLimit,
    PhysicalNestedLoopJoin,
    PhysicalOperator,
    PhysicalProject,
    PhysicalSort,
    PhysicalTableScan,
)
from .predicates import (
    combine_conjuncts,
    orient_equi_condition,
    resolve_table_index,
    split_conjuncts,
    touched_tables,
)
from .protocols import CardinalityEstimator, CostModel

#: Por encima de este número de tablas en una cadena de INNER JOIN, la
#: reordenación deja de explorar las `n!` ordenaciones (vía DP) y pasa a la
#: heurística greedy — `O(n · 2ⁿ)` con 12 tablas ya son ~49k combinaciones
#: de subconjunto/tabla añadida, un límite razonable sin volverse
#: impracticable (ver benchmark en el README, que mide el punto real donde
#: el coste de planificar empieza a dominar).
MAX_DP_TABLES = 12

#: Selectividad usada para un filtro sobre una relación ya derivada (tras un
#: `GROUP BY`/`HAVING`, o el `WHERE` final tras una cadena con algún outer
#: join) de la que no se conservan estadísticas por columna. Mismo valor y
#: misma justificación que `SimpleCardinalityEstimator.DEFAULT_UNKNOWN_SELECTIVITY`.
_DERIVED_RELATION_FILTER_SELECTIVITY = 0.5

_INDEXABLE_OPERATORS = frozenset(
    {
        BinaryOperator.EQ,
        BinaryOperator.NEQ,
        BinaryOperator.LT,
        BinaryOperator.LE,
        BinaryOperator.GT,
        BinaryOperator.GE,
    }
)


def build_physical_plan(
    node: LogicalOperator,
    catalog: Catalog,
    cost_model: CostModel,
    cardinality_estimator: CardinalityEstimator,
) -> PhysicalOperator:
    """Punto de entrada: recorre el plan lógico y construye el físico
    equivalente, tomando en cada nodo la decisión guiada por coste que le
    corresponde (ver docstring del módulo)."""

    if isinstance(node, LogicalScan | LogicalJoin):
        return _plan_relational(node, None, catalog, cost_model, cardinality_estimator)

    if isinstance(node, LogicalFilter):
        if isinstance(node.input, LogicalScan | LogicalJoin):
            return _plan_relational(
                node.input, node.predicate, catalog, cost_model, cardinality_estimator
            )
        # HAVING (filtro sobre un LogicalAggregate) u otro filtro sobre una
        # relación ya derivada: sin catálogo propio, se aplica como filtro
        # residual con selectividad por defecto (ver docstring del módulo).
        child = build_physical_plan(node.input, catalog, cost_model, cardinality_estimator)
        rows = child.estimated_rows * _DERIVED_RELATION_FILTER_SELECTIVITY
        cost = cost_model.cost_filter(child.estimated_cost, child.estimated_rows)
        return PhysicalFilter(child, node.predicate, rows, cost)

    if isinstance(node, LogicalDistinct):
        child = build_physical_plan(node.input, catalog, cost_model, cardinality_estimator)
        # Un DISTINCT es un hash aggregate sin funciones (agrupar por todas
        # las columnas proyectadas): misma forma de coste y misma reducción
        # determinista de cardinalidad que un GROUP BY sobre una relación
        # derivada — y reutilizar cost_aggregate evita añadir un método al
        # Protocol CostModel, que rompería implementaciones externas.
        output_rows = max(1.0, math.sqrt(child.estimated_rows))
        cost = cost_model.cost_aggregate(child.estimated_cost, child.estimated_rows, output_rows)
        return PhysicalDistinct(child, output_rows, cost)

    if isinstance(node, LogicalAggregate):
        child = build_physical_plan(node.input, catalog, cost_model, cardinality_estimator)
        return _plan_aggregate(child, node.group_by, node.aggregates, cost_model)

    if isinstance(node, LogicalProject):
        child = build_physical_plan(node.input, catalog, cost_model, cardinality_estimator)
        return PhysicalProject(child, node.items, child.estimated_rows, child.estimated_cost)

    if isinstance(node, LogicalSort):
        child = build_physical_plan(node.input, catalog, cost_model, cardinality_estimator)
        cost = cost_model.cost_sort(child.estimated_cost, child.estimated_rows)
        return PhysicalSort(child, node.order_by, child.estimated_rows, cost)

    if isinstance(node, LogicalLimit):
        child = build_physical_plan(node.input, catalog, cost_model, cardinality_estimator)
        rows = child.estimated_rows
        if node.limit is not None:
            rows = min(rows, float(node.limit))
        return PhysicalLimit(child, node.limit, node.offset, rows, child.estimated_cost)

    raise UnsupportedQueryError(f"nodo de plan lógico no soportado: {type(node).__name__}")


# ---------------------------------------------------------------------------
# FROM/JOIN + WHERE: pushdown, reordenación y selección de método físico
# ---------------------------------------------------------------------------


def _plan_relational(
    node: LogicalScan | LogicalJoin,
    where: Expression | None,
    catalog: Catalog,
    cost_model: CostModel,
    cardinality_estimator: CardinalityEstimator,
) -> PhysicalOperator:
    tables, steps = _flatten(node)

    if len(tables) == 1:
        conjuncts = split_conjuncts(where)
        return _plan_scan(tables[0], conjuncts, catalog, cost_model, cardinality_estimator)

    if all(join_type is JoinType.INNER for join_type, _ in steps):
        pool: list[Expression] = list(split_conjuncts(where))
        for _, condition in steps:
            pool.extend(split_conjuncts(condition))
        return _optimize_join_order(tables, pool, catalog, cost_model, cardinality_estimator)

    return _plan_join_chain_no_reorder(
        tables, steps, where, catalog, cost_model, cardinality_estimator
    )


def _flatten(node: LogicalOperator) -> tuple[list[TableRef], list[tuple[JoinType, Expression]]]:
    """Aplana la cadena left-deep de `LogicalJoin` que produce siempre
    `logical.build_logical_plan` en la lista de tablas (en el orden
    escrito) y la lista paralela de `(join_type, on_condition)` con la que
    cada tabla (a partir de la segunda) se adjunta a las anteriores."""

    if isinstance(node, LogicalScan):
        return [node.table], []
    if isinstance(node, LogicalJoin) and isinstance(node.right, LogicalScan):
        tables, steps = _flatten(node.left)
        return tables + [node.right.table], steps + [(node.join_type, node.condition)]
    raise UnsupportedQueryError(
        "physical.py sólo planifica cadenas de join left-deep (la única forma que produce "
        "logical.build_logical_plan); se recibió un LogicalJoin con un subárbol compuesto "
        "en el lado derecho"
    )


def _plan_join_chain_no_reorder(
    tables: list[TableRef],
    steps: list[tuple[JoinType, Expression]],
    where: Expression | None,
    catalog: Catalog,
    cost_model: CostModel,
    cardinality_estimator: CardinalityEstimator,
) -> PhysicalOperator:
    current = _plan_scan(tables[0], (), catalog, cost_model, cardinality_estimator)
    accumulated_tables = [tables[0]]
    for table, (join_type, condition) in zip(tables[1:], steps, strict=True):
        right = _plan_scan(table, (), catalog, cost_model, cardinality_estimator)
        current = _choose_physical_join(
            current,
            accumulated_tables,
            right,
            table,
            condition,
            join_type,
            catalog,
            cost_model,
            cardinality_estimator,
        )
        accumulated_tables.append(table)

    if where is None:
        return current
    rows = current.estimated_rows * _DERIVED_RELATION_FILTER_SELECTIVITY
    cost = cost_model.cost_filter(current.estimated_cost, current.estimated_rows)
    return PhysicalFilter(current, where, rows, cost)


# ---------------------------------------------------------------------------
# Pool de conjuntos + regla de aplicabilidad
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Conjunct:
    expression: Expression
    touched_mask: int


def _build_pool(
    pool: list[Expression], tables: list[TableRef], catalog: Catalog
) -> list[_Conjunct]:
    conjuncts = []
    for expression in pool:
        mask = 0
        for index in touched_tables(expression, tables, catalog):
            mask |= 1 << index
        conjuncts.append(_Conjunct(expression, mask))
    return conjuncts


def _applicable_at(conjuncts: list[_Conjunct], prev: int, t: int) -> list[Expression]:
    """Conjuntos que se aplican exactamente al añadir la tabla `t` sobre el
    acumulado `prev` (`prev` NO incluye todavía el bit de `t`; `prev = 0`
    para el caso base, el scan de `t` en solitario).

    Demostración de que cada conjunto se aplica en exactamente un paso, para
    cualquier orden de inserción `o = (t_1, ..., t_n)`: sea `T` el conjunto
    de tablas que toca un conjunto dado y `t_k` la ÚLTIMA de ellas en
    aparecer en `o`. En el paso que inserta `t_k` (`prev = {t_1,...,t_{k-1}}`),
    el subconjunto acumulado tras insertar es `{t_1, ..., t_k}` ⊇ `T` (todas
    las demás tablas de `T` ya están, por ser `t_k` la última) y `t_k ∈ T`,
    así que se aplica ahí. En cualquier paso anterior `j < k` con `t_j ∈ T`,
    el subconjunto acumulado tras insertar es `{t_1, ..., t_j}`, que NO
    contiene a `t_k` (aún no insertada) — así que `T ⊄ subconjunto` y no se
    aplica todavía. Por tanto se aplica en ese único paso, nunca antes,
    nunca después ni dos veces — **siempre que `prev` sea el acumulado
    EXACTO antes de insertar `t`**, no el subconjunto ya completo: un
    conjunto de una sola tabla (`touched = {t}`) satisface trivialmente
    `touched ⊆ subset` para CUALQUIER subconjunto que contenga a `t`, así
    que si se comprobase contra el subconjunto ya completo (en vez de
    contra `prev`) se volvería a "aplicar" cada vez que `t` se une a un
    acumulado, además de en su propio scan base — doble contabilización
    real, detectada por `test_properties.py` antes de publicar (ver
    `../PROGRESS.md`). La condición `touched_mask == bit and prev != 0` de
    abajo excluye exactamente ese caso: un conjunto puramente de `t` sólo
    cuenta en su scan base (`prev == 0`), nunca al unirse después.
    """

    bit = 1 << t
    subset = prev | bit
    applicable = []
    for c in conjuncts:
        if not (c.touched_mask & bit):
            continue
        if (c.touched_mask & subset) != c.touched_mask:
            continue
        if c.touched_mask == bit and prev != 0:
            continue
        applicable.append(c.expression)
    return applicable


def _optimize_join_order(
    tables: list[TableRef],
    pool: list[Expression],
    catalog: Catalog,
    cost_model: CostModel,
    cardinality_estimator: CardinalityEstimator,
) -> PhysicalOperator:
    conjuncts = _build_pool(pool, tables, catalog)
    # Un conjunto que no toca ninguna tabla (`ON 1 = 0`, `NULL IS NOT NULL`)
    # no tiene paso de inserción en el que "aplicarse" según la regla de
    # `_applicable_at` — antes se perdía en silencio y el join se volvía un
    # producto cartesiano. Es constante para todas las filas: se aplica una
    # vez como filtro sobre el resultado del join (en un INNER join es
    # equivalente a aplicarlo en cualquier paso).
    constants = [c.expression for c in conjuncts if c.touched_mask == 0]
    conjuncts = [c for c in conjuncts if c.touched_mask != 0]
    if len(tables) <= MAX_DP_TABLES:
        plan = _dp_join_order(tables, conjuncts, catalog, cost_model, cardinality_estimator)
    else:
        plan = _greedy_join_order(tables, conjuncts, catalog, cost_model, cardinality_estimator)
    predicate = combine_conjuncts(constants)
    if predicate is None:
        return plan
    rows = plan.estimated_rows * _DERIVED_RELATION_FILTER_SELECTIVITY
    cost = cost_model.cost_filter(plan.estimated_cost, plan.estimated_rows)
    return PhysicalFilter(plan, predicate, rows, cost)


def _subsets_of_size(n: int, size: int) -> Iterator[int]:
    for combo in itertools.combinations(range(n), size):
        mask = 0
        for index in combo:
            mask |= 1 << index
        yield mask


def _dp_join_order(
    tables: list[TableRef],
    conjuncts: list[_Conjunct],
    catalog: Catalog,
    cost_model: CostModel,
    cardinality_estimator: CardinalityEstimator,
) -> PhysicalOperator:
    n = len(tables)
    base = [
        _plan_scan(
            tables[i],
            _applicable_at(conjuncts, 0, i),
            catalog,
            cost_model,
            cardinality_estimator,
        )
        for i in range(n)
    ]
    dp: dict[int, PhysicalOperator] = {1 << i: base[i] for i in range(n)}

    for size in range(2, n + 1):
        for subset in _subsets_of_size(n, size):
            best: PhysicalOperator | None = None
            for t in range(n):
                bit = 1 << t
                if not (subset & bit):
                    continue
                prev = subset ^ bit
                if prev == 0:
                    continue
                applicable = _applicable_at(conjuncts, prev, t)
                candidate = _choose_physical_join(
                    dp[prev],
                    _tables_in_mask(tables, prev),
                    base[t],
                    tables[t],
                    combine_conjuncts(applicable),
                    JoinType.INNER,
                    catalog,
                    cost_model,
                    cardinality_estimator,
                )
                if best is None or candidate.estimated_cost < best.estimated_cost:
                    best = candidate
            assert best is not None
            dp[subset] = best

    return dp[(1 << n) - 1]


def _greedy_join_order(
    tables: list[TableRef],
    conjuncts: list[_Conjunct],
    catalog: Catalog,
    cost_model: CostModel,
    cardinality_estimator: CardinalityEstimator,
) -> PhysicalOperator:
    n = len(tables)
    base = [
        _plan_scan(
            tables[i],
            _applicable_at(conjuncts, 0, i),
            catalog,
            cost_model,
            cardinality_estimator,
        )
        for i in range(n)
    ]
    start = min(range(n), key=lambda i: (base[i].estimated_cost, i))
    current = base[start]
    subset = 1 << start
    remaining = [i for i in range(n) if i != start]

    while remaining:
        best_t: int | None = None
        best_candidate: PhysicalOperator | None = None
        for t in remaining:
            applicable = _applicable_at(conjuncts, subset, t)
            candidate = _choose_physical_join(
                current,
                _tables_in_mask(tables, subset),
                base[t],
                tables[t],
                combine_conjuncts(applicable),
                JoinType.INNER,
                catalog,
                cost_model,
                cardinality_estimator,
            )
            if best_candidate is None or candidate.estimated_cost < best_candidate.estimated_cost:
                best_candidate, best_t = candidate, t
        assert best_candidate is not None and best_t is not None
        current = best_candidate
        subset |= 1 << best_t
        remaining.remove(best_t)

    return current


def _tables_in_mask(tables: list[TableRef], mask: int) -> list[TableRef]:
    return [tables[i] for i in range(len(tables)) if mask & (1 << i)]


# ---------------------------------------------------------------------------
# Método físico de join
# ---------------------------------------------------------------------------


def _choose_physical_join(
    left: PhysicalOperator,
    left_tables: list[TableRef],
    right: PhysicalOperator,
    right_table: TableRef,
    condition: Expression | None,
    join_type: JoinType,
    catalog: Catalog,
    cost_model: CostModel,
    cardinality_estimator: CardinalityEstimator,
) -> PhysicalOperator:
    resolution_tables = [*left_tables, right_table]
    left_indices = frozenset(range(len(left_tables)))
    new_table_stats = catalog.get(right_table.name)

    # Se clasifica CADA conjunto de este paso por separado (no sólo el
    # primero reconocido como igualdad) — necesario para que la
    # cardinalidad resultante sea una función pura del subconjunto de
    # tablas y no dependa de en qué paso concreto se aplicó cada condición
    # (ver el docstring de `CardinalityEstimator.estimate_join_cardinality`
    # para el porqué exacto: si sólo la primera igualdad contase, dos
    # órdenes de inserción distintas que llegasen al mismo subconjunto
    # podrían estimar cardinalidades distintas, rompiendo la subestructura
    # óptima de la que depende `_dp_join_order`).
    equi_pairs: list[tuple[ColumnRef, ColumnRef]] = []
    equi_sources: list[Expression] = []
    non_equi_conjuncts: list[Expression] = []
    if condition is not None:
        for conjunct in split_conjuncts(condition):
            oriented = orient_equi_condition(conjunct, left_indices, resolution_tables, catalog)
            if oriented is not None:
                equi_pairs.append(oriented)
                equi_sources.append(conjunct)
            else:
                non_equi_conjuncts.append(conjunct)

    # Dos conjuntos DISTINTOS que resultan ser la MISMA igualdad (p. ej. la
    # misma condición repetida en el `ON` y, redundantemente, en el
    # `WHERE`) no son dos restricciones independientes — contarla dos veces
    # dividiría la cardinalidad por su `distinct_count` dos veces,
    # sobrestimando la selectividad real. Se deduplica por igualdad
    # estructural del par ya orientado antes de pasarlo al estimador.
    unique_equi_pairs = list(dict.fromkeys(equi_pairs))
    unique_non_equi_conjuncts = list(dict.fromkeys(non_equi_conjuncts))

    # `distinct_count` crudo (del catálogo, sin escalar por filtros ya
    # aplicados) de cada lado de cada igualdad — ver el docstring de
    # `CardinalityEstimator.estimate_join_cardinality` para el porqué de no
    # usar el `distinct_count` de una tabla ya filtrada.
    equi_join_distinct_counts: list[tuple[float, float]] = []
    for accumulated_column, new_column in unique_equi_pairs:
        accumulated_index = resolve_table_index(accumulated_column, resolution_tables, catalog)
        accumulated_stats = catalog.get(resolution_tables[accumulated_index].name)
        accumulated_column_stats = accumulated_stats.columns.get(accumulated_column.name)
        distinct_left = (
            float(accumulated_column_stats.distinct_count)
            if accumulated_column_stats is not None and accumulated_column_stats.distinct_count > 0
            else accumulated_stats.row_count
        )

        new_column_stats = new_table_stats.columns.get(new_column.name)
        distinct_right = (
            float(new_column_stats.distinct_count)
            if new_column_stats is not None and new_column_stats.distinct_count > 0
            else new_table_stats.row_count
        )
        equi_join_distinct_counts.append((distinct_left, distinct_right))

    output_rows = cardinality_estimator.estimate_join_cardinality(
        left.estimated_rows,
        right.estimated_rows,
        equi_join_distinct_counts,
        unrecognized_conjunct_count=len(unique_non_equi_conjuncts),
    )

    candidates: list[PhysicalOperator] = []

    nested_loop_cost = cost_model.cost_nested_loop_join(
        left.estimated_cost, left.estimated_rows, right.estimated_cost, output_rows
    )
    candidates.append(
        PhysicalNestedLoopJoin(left, right, join_type, condition, output_rows, nested_loop_cost)
    )

    if equi_pairs:
        # La primera igualdad reconocida (orden determinista: el de
        # `split_conjuncts`) se usa como clave de la tabla hash; cualquier
        # otra condición — incluida una segunda igualdad — se evalúa como
        # filtro residual encima del join, igual que en un hash join real
        # con más de una condición de igualdad.
        primary_left, primary_right = equi_pairs[0]
        hash_condition = BinaryOp(
            left=primary_left, operator=BinaryOperator.EQ, right=primary_right
        )
        residual_conjuncts = equi_sources[1:] + non_equi_conjuncts
        hash_build_cost = cost_model.cost_hash_join(
            left.estimated_cost,
            left.estimated_rows,
            right.estimated_cost,
            right.estimated_rows,
            output_rows,
        )
        hash_candidate: PhysicalOperator = PhysicalHashJoin(
            left, right, join_type, hash_condition, output_rows, hash_build_cost
        )
        if residual_conjuncts:
            residual_predicate = combine_conjuncts(residual_conjuncts)
            assert residual_predicate is not None
            residual_cost = cost_model.cost_filter(
                hash_candidate.estimated_cost, hash_candidate.estimated_rows
            )
            hash_candidate = PhysicalFilter(
                hash_candidate, residual_predicate, output_rows, residual_cost
            )
        candidates.append(hash_candidate)

    return min(candidates, key=lambda c: c.estimated_cost)


# ---------------------------------------------------------------------------
# Scan: full scan vs. índice
# ---------------------------------------------------------------------------


def _plan_scan(
    table: TableRef,
    conjuncts: Sequence[Expression],
    catalog: Catalog,
    cost_model: CostModel,
    cardinality_estimator: CardinalityEstimator,
) -> PhysicalOperator:
    stats = catalog.get(table.name)
    predicate = combine_conjuncts(list(conjuncts))
    if predicate is not None:
        selectivity = cardinality_estimator.estimate_selectivity(predicate, stats)
        filtered_rows = stats.row_count * selectivity
    else:
        filtered_rows = float(stats.row_count)

    full_scan_cost = cost_model.cost_table_scan(stats.row_count)
    best: PhysicalOperator
    if predicate is not None:
        best = PhysicalFilter(
            PhysicalTableScan(table, float(stats.row_count), full_scan_cost),
            predicate,
            filtered_rows,
            cost_model.cost_filter(full_scan_cost, stats.row_count),
        )
    else:
        best = PhysicalTableScan(table, float(stats.row_count), full_scan_cost)

    for index in stats.indexes:
        candidate = _plan_index_scan(
            table,
            index.name,
            index.columns[0],
            list(conjuncts),
            stats,
            filtered_rows,
            cost_model,
            cardinality_estimator,
        )
        if candidate is not None and candidate.estimated_cost < best.estimated_cost:
            best = candidate

    return best


def _plan_index_scan(
    table: TableRef,
    index_name: str,
    leading_column: str,
    conjuncts: list[Expression],
    stats: TableStatistics,
    filtered_rows: float,
    cost_model: CostModel,
    cardinality_estimator: CardinalityEstimator,
) -> PhysicalOperator | None:
    indexable = [c for c in conjuncts if _matches_index_column(c, leading_column)]
    if not indexable:
        return None
    residual = [c for c in conjuncts if c not in indexable]

    indexable_predicate = combine_conjuncts(indexable)
    assert indexable_predicate is not None
    matched_rows = stats.row_count * cardinality_estimator.estimate_selectivity(
        indexable_predicate, stats
    )
    index_cost = cost_model.cost_index_scan(matched_rows, stats.row_count)
    node: PhysicalOperator = PhysicalIndexScan(
        table, index_name, indexable_predicate, matched_rows, index_cost
    )

    if residual:
        residual_predicate = combine_conjuncts(residual)
        assert residual_predicate is not None
        node = PhysicalFilter(
            node,
            residual_predicate,
            filtered_rows,
            cost_model.cost_filter(index_cost, matched_rows),
        )
    return node


def _matches_index_column(expr: Expression, column_name: str) -> bool:
    if not isinstance(expr, BinaryOp) or expr.operator not in _INDEXABLE_OPERATORS:
        return False
    left, right = expr.left, expr.right
    if isinstance(left, ColumnRef) and left.name == column_name and isinstance(right, Literal):
        return True
    return isinstance(right, ColumnRef) and right.name == column_name and isinstance(left, Literal)


# ---------------------------------------------------------------------------
# Agregación
# ---------------------------------------------------------------------------


def _plan_aggregate(
    child: PhysicalOperator,
    group_by: tuple[Expression, ...],
    aggregates: tuple[FunctionCall, ...],
    cost_model: CostModel,
) -> PhysicalOperator:
    if not group_by:
        output_rows = 1.0
    else:
        # Sin estadísticas por columna de una relación ya derivada (el hijo
        # puede ser el resultado de varios joins), se usa una reducción
        # moderada y determinista en vez de inventar una cardinalidad de
        # grupos exacta sin base real. Extensión razonable pendiente,
        # comunicada aparte: resolver cada columna de GROUP BY contra el
        # catálogo cuando el agregado va directo sobre un único scan.
        output_rows = max(1.0, math.sqrt(child.estimated_rows))
    cost = cost_model.cost_aggregate(child.estimated_cost, child.estimated_rows, output_rows)
    return PhysicalHashAggregate(child, group_by, aggregates, output_rows, cost)
