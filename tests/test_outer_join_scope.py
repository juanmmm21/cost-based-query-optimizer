"""Alcance documentado en `physical.py`: una cadena de joins con algún
`LEFT`/`RIGHT`/`FULL OUTER JOIN` no se reordena ni recibe pushdown de
predicados, para no cambiar la semántica del outer join — sólo se elige
método físico por coste, y el `WHERE` final se aplica como un filtro
residual único encima de todo."""

from support import catalog, col, eq, eq_columns, inner_join, outer_join, ref, star_select, table

from cost_based_query_optimizer.models import JoinType, TableRef
from cost_based_query_optimizer.pipeline import optimize
from cost_based_query_optimizer.plan import (
    PhysicalFilter,
    PhysicalHashJoin,
    PhysicalNestedLoopJoin,
    PhysicalProject,
    PhysicalTableScan,
)


def test_left_join_chain_preserves_written_order_even_when_reordering_would_be_cheaper() -> None:
    # `small` es mucho más barata que `large`: si esto fuera INNER JOIN, el
    # optimizador la pondría primero. Al ser LEFT JOIN, debe conservar el
    # orden escrito (large primero, tal como aparece en el FROM).
    large = TableRef("large", "l")
    small = TableRef("small", "s")
    cat = catalog(
        large=table(1_000_000, {"id": col(1_000_000), "small_id": col(10)}),
        small=table(10, {"id": col(10)}),
    )
    select = star_select(
        large,
        joins=(outer_join(JoinType.LEFT, small, eq_columns(ref("small_id", "l"), ref("id", "s"))),),
    )
    plan = optimize(select, cat).physical_plan
    assert isinstance(plan, PhysicalProject)
    join_node = plan.input
    assert isinstance(join_node, PhysicalHashJoin | PhysicalNestedLoopJoin)
    assert join_node.join_type is JoinType.LEFT
    # El lado izquierdo del join físico debe ser el scan de `large` (orden
    # escrito), no el de `small` aunque fuese más barato empezar por ahí.
    left_side = join_node.left
    while not isinstance(left_side, PhysicalTableScan):
        left_side = left_side.input  # type: ignore[attr-defined]
    assert left_side.table.effective_name == "l"


def test_mixed_inner_and_outer_chain_disables_reordering_for_the_whole_chain() -> None:
    a = TableRef("a")
    b = TableRef("b")
    c = TableRef("c")
    cat = catalog(
        a=table(500_000, {"id": col(500_000), "b_id": col(10)}),
        b=table(10, {"id": col(10), "c_id": col(5)}),
        c=table(5, {"id": col(5)}),
    )
    select = star_select(
        a,
        joins=(
            inner_join(b, eq_columns(ref("b_id", "a"), ref("id", "b"))),
            outer_join(JoinType.LEFT, c, eq_columns(ref("c_id", "b"), ref("id", "c"))),
        ),
    )
    plan = optimize(select, cat).physical_plan
    assert isinstance(plan, PhysicalProject)
    outer_join_node = plan.input
    assert isinstance(outer_join_node, PhysicalHashJoin | PhysicalNestedLoopJoin)
    assert outer_join_node.join_type is JoinType.LEFT
    inner_join_node = outer_join_node.left
    assert isinstance(inner_join_node, PhysicalHashJoin | PhysicalNestedLoopJoin)
    assert inner_join_node.join_type is JoinType.INNER


def test_where_after_outer_join_is_a_single_residual_filter_not_pushed_down() -> None:
    orders = TableRef("orders", "o")
    customers = TableRef("customers", "c")
    cat = catalog(
        orders=table(1000, {"id": col(1000), "customer_id": col(100)}),
        customers=table(100, {"id": col(100), "country": col(5)}),
    )
    select = star_select(
        orders,
        joins=(
            outer_join(
                JoinType.LEFT, customers, eq_columns(ref("customer_id", "o"), ref("id", "c"))
            ),
        ),
        where=eq(ref("country", "c"), "ES"),
    )
    plan = optimize(select, cat).physical_plan
    assert isinstance(plan, PhysicalProject)
    assert isinstance(plan.input, PhysicalFilter)
    join_node = plan.input.input
    assert isinstance(join_node, PhysicalHashJoin | PhysicalNestedLoopJoin)
    assert join_node.join_type is JoinType.LEFT
