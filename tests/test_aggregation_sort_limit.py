from support import catalog, col, ref, table

from cost_based_query_optimizer.models import (
    FunctionCall,
    OrderByItem,
    OrderDirection,
    SelectItem,
    SelectStatement,
    Star,
    TableRef,
)
from cost_based_query_optimizer.pipeline import optimize
from cost_based_query_optimizer.plan import (
    PhysicalHashAggregate,
    PhysicalLimit,
    PhysicalProject,
    PhysicalSort,
    PhysicalTableScan,
)


def test_group_by_produces_hash_aggregate_under_project() -> None:
    cat = catalog(orders=table(1000, {"customer_id": col(100)}))
    select = SelectStatement(
        columns=(
            SelectItem(ref("customer_id"), alias="customer_id"),
            SelectItem(FunctionCall(name="COUNT", star_argument=True), alias="total"),
        ),
        from_table=TableRef("orders"),
        group_by=(ref("customer_id"),),
    )
    plan = optimize(select, cat).physical_plan
    assert isinstance(plan, PhysicalProject)
    assert isinstance(plan.input, PhysicalHashAggregate)
    assert plan.input.group_by == (ref("customer_id"),)
    assert len(plan.input.aggregates) == 1
    assert plan.input.aggregates[0].name == "COUNT"


def test_global_aggregate_without_group_by_estimates_single_row() -> None:
    cat = catalog(orders=table(1000, {"id": col(1000)}))
    select = SelectStatement(
        columns=(SelectItem(FunctionCall(name="COUNT", star_argument=True), alias="total"),),
        from_table=TableRef("orders"),
    )
    plan = optimize(select, cat).physical_plan
    assert isinstance(plan, PhysicalProject)
    assert isinstance(plan.input, PhysicalHashAggregate)
    assert plan.input.estimated_rows == 1.0


def test_query_without_aggregates_or_group_by_has_no_aggregate_node() -> None:
    cat = catalog(orders=table(1000, {"id": col(1000)}))
    select = SelectStatement(columns=(SelectItem(Star()),), from_table=TableRef("orders"))
    plan = optimize(select, cat).physical_plan
    assert isinstance(plan, PhysicalProject)
    assert isinstance(plan.input, PhysicalTableScan)


def test_order_by_wraps_project_in_sort() -> None:
    cat = catalog(orders=table(1000, {"id": col(1000)}))
    select = SelectStatement(
        columns=(SelectItem(Star()),),
        from_table=TableRef("orders"),
        order_by=(OrderByItem(ref("id"), OrderDirection.DESC),),
    )
    plan = optimize(select, cat).physical_plan
    assert isinstance(plan, PhysicalSort)
    assert isinstance(plan.input, PhysicalProject)
    assert plan.order_by[0].direction is OrderDirection.DESC


def test_limit_caps_estimated_rows() -> None:
    cat = catalog(orders=table(1000, {"id": col(1000)}))
    select = SelectStatement(columns=(SelectItem(Star()),), from_table=TableRef("orders"), limit=10)
    plan = optimize(select, cat).physical_plan
    assert isinstance(plan, PhysicalLimit)
    assert plan.limit == 10
    assert plan.estimated_rows == 10.0


def test_limit_does_not_increase_rows_when_larger_than_input() -> None:
    cat = catalog(orders=table(5, {"id": col(5)}))
    select = SelectStatement(
        columns=(SelectItem(Star()),), from_table=TableRef("orders"), limit=1000
    )
    plan = optimize(select, cat).physical_plan
    assert isinstance(plan, PhysicalLimit)
    assert plan.estimated_rows == 5.0


def test_full_clause_order_limit_over_sort_over_project() -> None:
    cat = catalog(orders=table(1000, {"id": col(1000)}))
    select = SelectStatement(
        columns=(SelectItem(Star()),),
        from_table=TableRef("orders"),
        order_by=(OrderByItem(ref("id")),),
        limit=5,
    )
    plan = optimize(select, cat).physical_plan
    assert isinstance(plan, PhysicalLimit)
    assert isinstance(plan.input, PhysicalSort)
    assert isinstance(plan.input.input, PhysicalProject)
