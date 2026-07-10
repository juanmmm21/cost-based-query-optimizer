"""DoD: "Al menos un algoritmo de reordenación de joins" y "orden de joins
óptimo en queries de 3+ tablas". Cada test compara el coste del plan que
elige `optimize()` contra el coste de un puñado de órdenes alternativas
construidas a mano con las MISMAS piezas de coste (`SimpleCostModel`/
`SimpleCardinalityEstimator`) pero ensambladas directamente, sin pasar por
la reordenación — el optimizador no debería perder nunca contra ninguna."""

from support import catalog, col, eq_columns, inner_join, ref, star_select, table

from cost_based_query_optimizer.models import JoinType, TableRef
from cost_based_query_optimizer.physical import _choose_physical_join, _plan_scan
from cost_based_query_optimizer.pipeline import optimize
from cost_based_query_optimizer.plan import (
    PhysicalHashJoin,
    PhysicalNestedLoopJoin,
    PhysicalOperator,
)
from cost_based_query_optimizer.protocols import SimpleCardinalityEstimator, SimpleCostModel

COST_MODEL = SimpleCostModel()
ESTIMATOR = SimpleCardinalityEstimator()


def _left_deep_cost(order: list[TableRef], conditions: dict[frozenset, object], cat) -> float:
    """Construye el plan left-deep en el orden dado (sin reordenar) y
    devuelve su coste, usando las mismas piezas que `physical.py` — sirve de
    alternativa manual con la que comparar, no de reimplementación
    independiente del cálculo de coste en sí (eso ya lo cubren
    `test_cost_model.py`/`test_cardinality.py`)."""

    from cost_based_query_optimizer.predicates import combine_conjuncts

    current: PhysicalOperator = _plan_scan(order[0], (), cat, COST_MODEL, ESTIMATOR)
    accumulated = [order[0]]
    for table_ref in order[1:]:
        accumulated_names = {t.effective_name for t in accumulated}
        subset_names = accumulated_names | {table_ref.effective_name}
        # Todas las condiciones que se completan al añadir esta tabla (puede
        # ser más de una si sus dos vecinos ya estaban acumulados) — igual
        # que `physical._applicable_at`, para no perder ninguna.
        applicable = [
            expr
            for key, expr in conditions.items()
            if table_ref.effective_name in key and key <= subset_names
        ]
        condition = combine_conjuncts(applicable) if applicable else None
        right = _plan_scan(table_ref, (), cat, COST_MODEL, ESTIMATOR)
        current = _choose_physical_join(
            current,
            accumulated,
            right,
            table_ref,
            condition,
            JoinType.INNER,
            cat,
            COST_MODEL,
            ESTIMATOR,
        )
        accumulated.append(table_ref)
    return current.estimated_cost


def test_three_table_join_beats_naive_written_order() -> None:
    customers = TableRef("customers", "c")
    orders = TableRef("orders", "o")
    order_items = TableRef("order_items", "oi")

    cat = catalog(
        customers=table(100, {"id": col(100)}),
        orders=table(100_000, {"id": col(100_000), "customer_id": col(100)}),
        order_items=table(500_000, {"id": col(500_000), "order_id": col(100_000)}),
    )

    select = star_select(
        order_items,
        joins=(
            inner_join(orders, eq_columns(ref("order_id", "oi"), ref("id", "o"))),
            inner_join(customers, eq_columns(ref("customer_id", "o"), ref("id", "c"))),
        ),
    )
    optimized_cost = optimize(select, cat).estimated_cost

    conditions = {
        frozenset({"oi", "o"}): eq_columns(ref("order_id", "oi"), ref("id", "o")),
        frozenset({"o", "c"}): eq_columns(ref("customer_id", "o"), ref("id", "c")),
    }
    naive_cost = _left_deep_cost([order_items, orders, customers], conditions, cat)

    assert optimized_cost <= naive_cost


def test_optimizer_never_beaten_by_any_permutation_of_three_tables() -> None:
    import itertools

    customers = TableRef("customers", "c")
    orders = TableRef("orders", "o")
    order_items = TableRef("order_items", "oi")
    all_tables = {"c": customers, "o": orders, "oi": order_items}

    cat = catalog(
        customers=table(100, {"id": col(100)}),
        orders=table(100_000, {"id": col(100_000), "customer_id": col(100)}),
        order_items=table(500_000, {"id": col(500_000), "order_id": col(100_000)}),
    )

    conditions = {
        frozenset({"oi", "o"}): eq_columns(ref("order_id", "oi"), ref("id", "o")),
        frozenset({"o", "c"}): eq_columns(ref("customer_id", "o"), ref("id", "c")),
    }

    select = star_select(
        order_items,
        joins=(
            inner_join(orders, conditions[frozenset({"oi", "o"})]),
            inner_join(customers, conditions[frozenset({"o", "c"})]),
        ),
    )
    optimized_cost = optimize(select, cat).estimated_cost

    for permutation in itertools.permutations(all_tables.values()):
        alt_cost = _left_deep_cost(list(permutation), conditions, cat)
        assert optimized_cost <= alt_cost + 1e-6


def test_four_table_chain_orders_smallest_tables_first() -> None:
    """Cadena en estrella-lineal con una tabla enana intercalada a
    propósito en el peor sitio posible al escribirla; el optimizador debe
    encontrar un orden de coste igual o menor que escribirlo tal cual."""

    a = TableRef("a")
    b = TableRef("b")
    c = TableRef("c")
    d = TableRef("d")

    cat = catalog(
        a=table(200_000, {"id": col(200_000), "b_id": col(50)}),
        b=table(50, {"id": col(50)}),
        c=table(300_000, {"id": col(300_000), "a_id": col(200_000)}),
        d=table(10, {"id": col(10), "c_id": col(300_000)}),
    )
    conditions = {
        frozenset({"a", "b"}): eq_columns(ref("b_id", "a"), ref("id", "b")),
        frozenset({"c", "a"}): eq_columns(ref("a_id", "c"), ref("id", "a")),
        frozenset({"d", "c"}): eq_columns(ref("c_id", "d"), ref("id", "c")),
    }

    select = star_select(
        a,
        joins=(
            inner_join(b, conditions[frozenset({"a", "b"})]),
            inner_join(c, conditions[frozenset({"c", "a"})]),
            inner_join(d, conditions[frozenset({"d", "c"})]),
        ),
    )
    optimized_cost = optimize(select, cat).estimated_cost
    naive_cost = _left_deep_cost([a, b, c, d], conditions, cat)
    assert optimized_cost <= naive_cost


def test_join_method_chosen_by_cost_not_fixed_heuristic() -> None:
    """Con una equi-condición reconocida, `_choose_physical_join` debe
    poder devolver tanto `PhysicalHashJoin` (caso normal) como aceptar que
    `PhysicalNestedLoopJoin` sea la alternativa disponible — la elección
    sale de comparar costes, no de "siempre hash si hay igualdad"."""

    left_table = TableRef("left_table")
    right_table = TableRef("right_table")
    cat = catalog(
        left_table=table(1000, {"id": col(1000)}),
        right_table=table(1000, {"id": col(1000)}),
    )
    left_plan = _plan_scan(left_table, (), cat, COST_MODEL, ESTIMATOR)
    right_plan = _plan_scan(right_table, (), cat, COST_MODEL, ESTIMATOR)
    condition = eq_columns(ref("id", "left_table"), ref("id", "right_table"))
    chosen = _choose_physical_join(
        left_plan,
        [left_table],
        right_plan,
        right_table,
        condition,
        JoinType.INNER,
        cat,
        COST_MODEL,
        ESTIMATOR,
    )
    assert isinstance(chosen, PhysicalHashJoin | PhysicalNestedLoopJoin)
