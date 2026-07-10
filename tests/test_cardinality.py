import pytest
from support import and_, cmp, col, eq, ref, table

from cost_based_query_optimizer.models import (
    BinaryOperator,
    HistogramBucket,
    UnaryOp,
    UnaryOperator,
)
from cost_based_query_optimizer.protocols import SimpleCardinalityEstimator

ESTIMATOR = SimpleCardinalityEstimator()


def test_equality_selectivity_uses_distinct_count() -> None:
    stats = table(1000, {"status": col(10)})
    predicate = eq(ref("status"), "PENDING")
    assert ESTIMATOR.estimate_selectivity(predicate, stats) == pytest.approx(0.1)


def test_equality_selectivity_falls_back_without_column_stats() -> None:
    stats = table(1000, {})
    predicate = eq(ref("status"), "PENDING")
    expected = 1.0 / SimpleCardinalityEstimator.DEFAULT_DISTINCT_COUNT_FOR_EQUALITY
    assert ESTIMATOR.estimate_selectivity(predicate, stats) == pytest.approx(expected)


def test_neq_is_complement_of_equality() -> None:
    stats = table(1000, {"status": col(4)})
    eq_sel = ESTIMATOR.estimate_selectivity(eq(ref("status"), "A"), stats)
    neq_sel = ESTIMATOR.estimate_selectivity(cmp(ref("status"), BinaryOperator.NEQ, "A"), stats)
    assert neq_sel == pytest.approx(1.0 - eq_sel)


def test_and_multiplies_selectivities() -> None:
    stats = table(1000, {"a": col(10), "b": col(5)})
    combined = and_(eq(ref("a"), 1), eq(ref("b"), 1))
    assert ESTIMATOR.estimate_selectivity(combined, stats) == pytest.approx(0.1 * 0.2)


def test_or_uses_inclusion_exclusion() -> None:
    from cost_based_query_optimizer.models import BinaryOp

    stats = table(1000, {"a": col(10), "b": col(5)})
    expr = BinaryOp(left=eq(ref("a"), 1), operator=BinaryOperator.OR, right=eq(ref("b"), 1))
    p_a, p_b = 0.1, 0.2
    expected = p_a + p_b - p_a * p_b
    assert ESTIMATOR.estimate_selectivity(expr, stats) == pytest.approx(expected)


def test_not_is_complement() -> None:
    stats = table(1000, {"a": col(10)})
    expr = UnaryOp(operator=UnaryOperator.NOT, operand=eq(ref("a"), 1))
    assert ESTIMATOR.estimate_selectivity(expr, stats) == pytest.approx(0.9)


def test_range_without_histogram_uses_default() -> None:
    stats = table(1000, {"price": col(500)})
    predicate = cmp(ref("price"), BinaryOperator.GT, 100)
    assert ESTIMATOR.estimate_selectivity(predicate, stats) == pytest.approx(
        SimpleCardinalityEstimator.DEFAULT_RANGE_SELECTIVITY
    )


def test_range_with_histogram_computes_overlap_fraction() -> None:
    histogram = (
        HistogramBucket(lower_bound=0, upper_bound=100, row_count=500, distinct_count=100),
        HistogramBucket(lower_bound=100, upper_bound=200, row_count=500, distinct_count=100),
    )
    stats = table(1000, {"price": col(200, histogram=histogram)})
    # price > 150: cubre la mitad del segundo cubo (150..200 de 100..200) -> 250 filas / 1000
    predicate = cmp(ref("price"), BinaryOperator.GT, 150)
    assert ESTIMATOR.estimate_selectivity(predicate, stats) == pytest.approx(0.25)


def test_range_with_histogram_flips_literal_on_left() -> None:
    histogram = (
        HistogramBucket(lower_bound=0, upper_bound=100, row_count=500, distinct_count=100),
        HistogramBucket(lower_bound=100, upper_bound=200, row_count=500, distinct_count=100),
    )
    stats = table(1000, {"price": col(200, histogram=histogram)})
    from cost_based_query_optimizer.models import BinaryOp, Literal, LiteralType

    predicate = BinaryOp(
        left=Literal(150, LiteralType.INTEGER), operator=BinaryOperator.LT, right=ref("price")
    )
    assert ESTIMATOR.estimate_selectivity(predicate, stats) == pytest.approx(0.25)


def test_unknown_predicate_shape_uses_default() -> None:
    from cost_based_query_optimizer.models import Between, Literal, LiteralType

    stats = table(1000, {"price": col(200)})
    predicate = Between(
        operand=ref("price"),
        low=Literal(1, LiteralType.INTEGER),
        high=Literal(2, LiteralType.INTEGER),
    )
    assert ESTIMATOR.estimate_selectivity(predicate, stats) == pytest.approx(
        SimpleCardinalityEstimator.DEFAULT_UNKNOWN_SELECTIVITY
    )


def test_join_cardinality_cross_product_without_predicate() -> None:
    result = ESTIMATOR.estimate_join_cardinality(100, 50, [], unrecognized_conjunct_count=0)
    assert result == pytest.approx(100 * 50)


def test_join_cardinality_uses_containment_formula() -> None:
    result = ESTIMATOR.estimate_join_cardinality(
        1000, 50, [(50, 50)], unrecognized_conjunct_count=0
    )
    assert result == pytest.approx((1000 * 50) / 50)


def test_join_cardinality_uses_max_of_both_sides_symmetrically() -> None:
    # a.x (distinct=10) = b.y (distinct=50): el resultado no debe depender
    # de cuál de las dos tablas se considere "nueva" en este paso — ver el
    # docstring de `estimate_join_cardinality` (simetría necesaria para que
    # la reordenación de joins sea determinista y correcta).
    forward = ESTIMATOR.estimate_join_cardinality(
        1000, 50, [(10, 50)], unrecognized_conjunct_count=0
    )
    backward = ESTIMATOR.estimate_join_cardinality(
        50, 1000, [(50, 10)], unrecognized_conjunct_count=0
    )
    assert forward == pytest.approx((1000 * 50) / 50)
    assert backward == pytest.approx((50 * 1000) / 50)


def test_join_cardinality_multiple_equi_columns_each_reduce_independently() -> None:
    result = ESTIMATOR.estimate_join_cardinality(
        1000, 50, [(50, 50), (5, 5)], unrecognized_conjunct_count=0
    )
    assert result == pytest.approx((1000 * 50) / 50 / 5)


def test_join_cardinality_unrecognized_predicate_reduces_by_default_factor() -> None:
    result = ESTIMATOR.estimate_join_cardinality(100, 50, [], unrecognized_conjunct_count=1)
    assert result == pytest.approx(100 * 50 * SimpleCardinalityEstimator.DEFAULT_JOIN_SELECTIVITY)


def test_join_cardinality_two_unrecognized_predicates_compound() -> None:
    result = ESTIMATOR.estimate_join_cardinality(100, 50, [], unrecognized_conjunct_count=2)
    assert result == pytest.approx(
        100 * 50 * SimpleCardinalityEstimator.DEFAULT_JOIN_SELECTIVITY**2
    )


def test_selectivity_is_always_clamped_to_unit_interval() -> None:
    stats = table(1000, {"a": col(1), "b": col(1)})
    combined = and_(eq(ref("a"), 1), eq(ref("b"), 1))
    for _ in range(3):
        combined = and_(combined, eq(ref("a"), 1))
    result = ESTIMATOR.estimate_selectivity(combined, stats)
    assert 0.0 <= result <= 1.0
