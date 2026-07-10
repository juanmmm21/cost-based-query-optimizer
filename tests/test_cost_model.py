import pytest

from cost_based_query_optimizer.protocols import SimpleCostModel

MODEL = SimpleCostModel()


def test_table_scan_cost_grows_with_row_count() -> None:
    small = MODEL.cost_table_scan(100)
    large = MODEL.cost_table_scan(100_000)
    assert large > small


def test_table_scan_cost_never_zero_for_nonempty_table() -> None:
    assert MODEL.cost_table_scan(1) > 0


def test_index_scan_cheaper_than_table_scan_when_highly_selective() -> None:
    table_scan = MODEL.cost_table_scan(100_000)
    index_scan = MODEL.cost_index_scan(matched_rows=1, table_row_count=100_000)
    assert index_scan < table_scan


def test_index_scan_more_expensive_than_table_scan_when_barely_selective() -> None:
    table_scan = MODEL.cost_table_scan(100_000)
    index_scan = MODEL.cost_index_scan(matched_rows=90_000, table_row_count=100_000)
    assert index_scan > table_scan


def test_index_scan_cost_scales_with_matched_rows() -> None:
    small = MODEL.cost_index_scan(matched_rows=10, table_row_count=100_000)
    large = MODEL.cost_index_scan(matched_rows=10_000, table_row_count=100_000)
    assert large > small


def test_index_scan_has_minimum_cost_with_zero_matches() -> None:
    assert MODEL.cost_index_scan(matched_rows=0, table_row_count=100_000) > 0


def test_filter_adds_cost_proportional_to_input_rows() -> None:
    base = MODEL.cost_filter(input_cost=10.0, input_rows=0)
    with_rows = MODEL.cost_filter(input_cost=10.0, input_rows=1000)
    assert with_rows > base


def test_nested_loop_join_cost_scales_with_left_rows() -> None:
    small_left = MODEL.cost_nested_loop_join(
        left_cost=1.0, left_rows=10, right_cost=5.0, output_rows=10
    )
    large_left = MODEL.cost_nested_loop_join(
        left_cost=1.0, left_rows=10_000, right_cost=5.0, output_rows=10
    )
    assert large_left > small_left


def test_hash_join_cheaper_than_nested_loop_for_large_inputs() -> None:
    left_cost, left_rows = MODEL.cost_table_scan(100_000), 100_000.0
    right_cost, right_rows = MODEL.cost_table_scan(100_000), 100_000.0
    output_rows = 100_000.0
    nested_loop = MODEL.cost_nested_loop_join(left_cost, left_rows, right_cost, output_rows)
    hash_join = MODEL.cost_hash_join(left_cost, left_rows, right_cost, right_rows, output_rows)
    assert hash_join < nested_loop


def test_sort_cost_grows_faster_than_linear() -> None:
    cost_1k = MODEL.cost_sort(input_cost=0.0, input_rows=1_000)
    cost_1m = MODEL.cost_sort(input_cost=0.0, input_rows=1_000_000)
    # n log n: multiplicar filas x1000 debe multiplicar el coste más de x1000... no,
    # menos que x1000^2 pero más que x1000 (el log crece). Se verifica la cota inferior
    # (estrictamente mayor que un crecimiento puramente lineal) y la superior (mucho
    # menor que cuadrático).
    assert cost_1m > cost_1k * 1000
    assert cost_1m < cost_1k * 1000 * 1000


def test_sort_of_empty_input_has_no_extra_cost() -> None:
    assert MODEL.cost_sort(input_cost=5.0, input_rows=0) == pytest.approx(5.0)


def test_aggregate_cost_includes_input_and_output_rows() -> None:
    cheap = MODEL.cost_aggregate(input_cost=0.0, input_rows=100, output_rows=10)
    expensive = MODEL.cost_aggregate(input_cost=0.0, input_rows=100_000, output_rows=1_000)
    assert expensive > cheap
