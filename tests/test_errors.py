import pytest
from support import catalog, col, eq, eq_columns, inner_join, ref, star_select, table

from cost_based_query_optimizer.errors import (
    MissingStatisticsError,
    UnresolvedColumnError,
    UnsupportedQueryError,
)
from cost_based_query_optimizer.models import JoinType, TableRef
from cost_based_query_optimizer.physical import build_physical_plan
from cost_based_query_optimizer.pipeline import optimize
from cost_based_query_optimizer.plan import LogicalJoin, LogicalScan
from cost_based_query_optimizer.protocols import SimpleCardinalityEstimator, SimpleCostModel


def test_missing_statistics_error_for_unknown_table_in_from() -> None:
    cat = catalog(customers=table(10, {"id": col(10)}))
    select = star_select(TableRef("orders"))
    with pytest.raises(MissingStatisticsError) as excinfo:
        optimize(select, cat)
    assert excinfo.value.table_name == "orders"


def test_missing_statistics_error_for_unknown_table_in_join() -> None:
    cat = catalog(orders=table(10, {"id": col(10), "customer_id": col(5)}))
    select = star_select(
        TableRef("orders", "o"),
        joins=(
            inner_join(
                TableRef("customers", "c"), eq_columns(ref("customer_id", "o"), ref("id", "c"))
            ),
        ),
    )
    with pytest.raises(MissingStatisticsError) as excinfo:
        optimize(select, cat)
    assert excinfo.value.table_name == "customers"


def test_unresolved_column_error_for_ambiguous_unqualified_reference() -> None:
    cat = catalog(
        orders=table(10, {"id": col(10)}),
        customers=table(5, {"id": col(5)}),
    )
    select = star_select(
        TableRef("orders", "o"),
        joins=(inner_join(TableRef("customers", "c"), eq_columns(ref("id", "o"), ref("id", "c"))),),
        where=eq(ref("id"), 1),
    )
    with pytest.raises(UnresolvedColumnError):
        optimize(select, cat)


def test_unsupported_query_error_for_bushy_logical_join() -> None:
    cat = catalog(
        a=table(10, {"id": col(10)}),
        b=table(10, {"id": col(10)}),
        c=table(10, {"id": col(10)}),
        d=table(10, {"id": col(10)}),
    )
    # `logical.build_logical_plan` nunca produce este árbol (siempre
    # left-deep) — se construye a mano para verificar que `physical.py`
    # falla explícitamente en vez de planificar mal en silencio.
    bushy = LogicalJoin(
        left=LogicalJoin(
            left=LogicalScan(TableRef("a")),
            right=LogicalScan(TableRef("b")),
            join_type=JoinType.INNER,
            condition=eq_columns(ref("id", "a"), ref("id", "b")),
        ),
        right=LogicalJoin(
            left=LogicalScan(TableRef("c")),
            right=LogicalScan(TableRef("d")),
            join_type=JoinType.INNER,
            condition=eq_columns(ref("id", "c"), ref("id", "d")),
        ),
        join_type=JoinType.INNER,
        condition=eq_columns(ref("id", "b"), ref("id", "c")),
    )
    with pytest.raises(UnsupportedQueryError):
        build_physical_plan(bushy, cat, SimpleCostModel(), SimpleCardinalityEstimator())
