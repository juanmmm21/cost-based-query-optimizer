"""Planificación de `SELECT DISTINCT`: un `PhysicalDistinct` entre el
proyector y el orden, con cardinalidad reducida de forma determinista."""

from __future__ import annotations

from support import catalog, col, ref, table

from cost_based_query_optimizer.models import (
    OrderByItem,
    SelectItem,
    SelectStatement,
    Star,
    TableRef,
)
from cost_based_query_optimizer.pipeline import explain, optimize
from cost_based_query_optimizer.plan import (
    PhysicalDistinct,
    PhysicalProject,
    PhysicalSort,
)


def _distinct_select(**kwargs: object) -> SelectStatement:
    return SelectStatement(
        columns=(SelectItem(Star()),),
        from_table=TableRef("orders"),
        distinct=True,
        **kwargs,  # type: ignore[arg-type]
    )


def test_distinct_wraps_project() -> None:
    cat = catalog(orders=table(1000, {"id": col(1000)}))
    plan = optimize(_distinct_select(), cat).physical_plan
    assert isinstance(plan, PhysicalDistinct)
    assert isinstance(plan.input, PhysicalProject)


def test_sort_goes_above_distinct() -> None:
    # SQL deduplica la lista de salida y ordena el resultado deduplicado.
    cat = catalog(orders=table(1000, {"id": col(1000)}))
    plan = optimize(_distinct_select(order_by=(OrderByItem(ref("id")),)), cat).physical_plan
    assert isinstance(plan, PhysicalSort)
    assert isinstance(plan.input, PhysicalDistinct)
    assert isinstance(plan.input.input, PhysicalProject)


def test_distinct_reduces_estimated_rows_deterministically() -> None:
    cat = catalog(orders=table(10_000, {"id": col(10_000)}))
    first = optimize(_distinct_select(), cat).physical_plan
    second = optimize(_distinct_select(), cat).physical_plan
    assert first == second  # determinismo: mismo AST + stats => mismo plan
    assert isinstance(first, PhysicalDistinct)
    assert 1.0 <= first.estimated_rows < first.input.estimated_rows
    assert first.estimated_cost > first.input.estimated_cost


def test_non_distinct_plan_has_no_distinct_node() -> None:
    cat = catalog(orders=table(1000, {"id": col(1000)}))
    select = SelectStatement(columns=(SelectItem(Star()),), from_table=TableRef("orders"))
    plan = optimize(select, cat).physical_plan
    assert isinstance(plan, PhysicalProject)


def test_explain_renders_distinct_node() -> None:
    cat = catalog(orders=table(1000, {"id": col(1000)}))
    plan = optimize(_distinct_select(), cat).physical_plan
    assert "Distinct" in explain(plan)
