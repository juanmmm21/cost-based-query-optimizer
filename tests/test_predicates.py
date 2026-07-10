import pytest
from support import and_, catalog, col, eq, eq_columns, ref, table

from cost_based_query_optimizer.errors import UnresolvedColumnError
from cost_based_query_optimizer.models import BinaryOperator, TableRef
from cost_based_query_optimizer.predicates import (
    as_equi_columns,
    combine_conjuncts,
    orient_equi_condition,
    resolve_table_index,
    split_conjuncts,
    touched_tables,
)


def test_split_conjuncts_none_returns_empty() -> None:
    assert split_conjuncts(None) == ()


def test_split_conjuncts_single_expression_not_and() -> None:
    expr = eq(ref("x"), 1)
    assert split_conjuncts(expr) == (expr,)


def test_split_conjuncts_and_chain() -> None:
    a, b, c = eq(ref("x"), 1), eq(ref("y"), 2), eq(ref("z"), 3)
    chained = and_(a, b, c)
    assert split_conjuncts(chained) == (a, b, c)


def test_split_conjuncts_does_not_split_or() -> None:
    from cost_based_query_optimizer.models import BinaryOp

    or_expr = BinaryOp(left=eq(ref("x"), 1), operator=BinaryOperator.OR, right=eq(ref("y"), 2))
    assert split_conjuncts(or_expr) == (or_expr,)


def test_combine_conjuncts_empty_is_none() -> None:
    assert combine_conjuncts([]) is None


def test_combine_conjuncts_single_is_unwrapped() -> None:
    a = eq(ref("x"), 1)
    assert combine_conjuncts([a]) is a


def test_combine_conjuncts_roundtrips_with_split() -> None:
    a, b, c = eq(ref("x"), 1), eq(ref("y"), 2), eq(ref("z"), 3)
    combined = combine_conjuncts([a, b, c])
    assert combined is not None
    assert split_conjuncts(combined) == (a, b, c)


def test_resolve_table_index_qualified() -> None:
    tables = [TableRef("orders", "o"), TableRef("customers", "c")]
    cat = catalog(orders=table(10, {"id": col(10)}), customers=table(5, {"id": col(5)}))
    assert resolve_table_index(ref("id", "o"), tables, cat) == 0
    assert resolve_table_index(ref("id", "c"), tables, cat) == 1


def test_resolve_table_index_qualified_unknown_alias_raises() -> None:
    tables = [TableRef("orders", "o")]
    cat = catalog(orders=table(10, {"id": col(10)}))
    with pytest.raises(UnresolvedColumnError):
        resolve_table_index(ref("id", "missing"), tables, cat)


def test_resolve_table_index_unqualified_unique_match() -> None:
    tables = [TableRef("orders", "o"), TableRef("customers", "c")]
    cat = catalog(
        orders=table(10, {"order_id": col(10)}),
        customers=table(5, {"customer_id": col(5)}),
    )
    assert resolve_table_index(ref("order_id"), tables, cat) == 0


def test_resolve_table_index_unqualified_ambiguous_raises() -> None:
    tables = [TableRef("orders", "o"), TableRef("customers", "c")]
    cat = catalog(orders=table(10, {"id": col(10)}), customers=table(5, {"id": col(5)}))
    with pytest.raises(UnresolvedColumnError):
        resolve_table_index(ref("id"), tables, cat)


def test_resolve_table_index_unqualified_no_match_raises() -> None:
    tables = [TableRef("orders", "o")]
    cat = catalog(orders=table(10, {"id": col(10)}))
    with pytest.raises(UnresolvedColumnError):
        resolve_table_index(ref("nonexistent"), tables, cat)


def test_touched_tables_multi_table_expression() -> None:
    tables = [TableRef("orders", "o"), TableRef("customers", "c")]
    cat = catalog(
        orders=table(10, {"customer_id": col(10)}),
        customers=table(5, {"id": col(5)}),
    )
    expr = eq_columns(ref("customer_id", "o"), ref("id", "c"))
    assert touched_tables(expr, tables, cat) == frozenset({0, 1})


def test_touched_tables_single_table_expression() -> None:
    tables = [TableRef("orders", "o")]
    cat = catalog(orders=table(10, {"status": col(3)}))
    expr = eq(ref("status", "o"), "PENDING")
    assert touched_tables(expr, tables, cat) == frozenset({0})


def test_as_equi_columns_recognizes_direct_equality() -> None:
    left, right = ref("x", "a"), ref("y", "b")
    assert as_equi_columns(eq_columns(left, right)) == (left, right)


def test_as_equi_columns_rejects_comparison_with_literal() -> None:
    assert as_equi_columns(eq(ref("x"), 1)) is None


def test_orient_equi_condition_reorders_to_match_left_side() -> None:
    tables = [TableRef("orders", "o"), TableRef("customers", "c")]
    cat = catalog(
        orders=table(10, {"customer_id": col(10)}),
        customers=table(5, {"id": col(5)}),
    )
    # Escrito con el lado "nuevo" (customers, índice 1) primero.
    expr = eq_columns(ref("id", "c"), ref("customer_id", "o"))
    oriented = orient_equi_condition(expr, frozenset({0}), tables, cat)
    assert oriented == (ref("customer_id", "o"), ref("id", "c"))


def test_orient_equi_condition_returns_none_when_both_sides_on_same_side() -> None:
    tables = [TableRef("orders", "o"), TableRef("order_items", "oi")]
    cat = catalog(
        orders=table(10, {"id": col(10), "total": col(10)}),
        order_items=table(20, {"order_id": col(10)}),
    )
    # Ambas columnas pertenecen al lado izquierdo: no es una condición de join.
    expr = eq_columns(ref("id", "o"), ref("total", "o"))
    assert orient_equi_condition(expr, frozenset({0}), tables, cat) is None
