"""DoD: "Selección de índice vs full scan basada en coste estimado, no en
heurísticas fijas" y "uso de índice cuando existe y es selectivo, full scan
cuando no lo es"."""

from support import catalog, cmp, col, eq, index, ref, star_select, table

from cost_based_query_optimizer.models import BinaryOperator, TableRef
from cost_based_query_optimizer.pipeline import optimize
from cost_based_query_optimizer.plan import PhysicalFilter, PhysicalIndexScan, PhysicalTableScan


def _scan_under_project(result):
    node = result.physical_plan
    while not isinstance(node, PhysicalIndexScan | PhysicalTableScan | PhysicalFilter):
        node = node.input  # type: ignore[attr-defined]
    return node


def test_selective_equality_predicate_uses_index() -> None:
    cat = catalog(
        orders=table(
            100_000,
            {"id": col(100_000), "customer_id": col(100_000)},
            indexes=(index("idx_customer_id", "customer_id"),),
        )
    )
    select = star_select(TableRef("orders"), where=eq(ref("customer_id"), 42))
    result = optimize(select, cat)
    node = _scan_under_project(result)
    assert isinstance(node, PhysicalIndexScan)
    assert node.index_name == "idx_customer_id"


def test_unselective_equality_predicate_uses_full_scan() -> None:
    cat = catalog(
        orders=table(
            100_000,
            {"id": col(100_000), "status": col(3)},
            indexes=(index("idx_status", "status"),),
        )
    )
    select = star_select(TableRef("orders"), where=eq(ref("status"), "PENDING"))
    result = optimize(select, cat)
    node = _scan_under_project(result)
    assert isinstance(node, PhysicalFilter)
    assert isinstance(node.input, PhysicalTableScan)


def test_no_predicate_uses_bare_full_scan() -> None:
    cat = catalog(orders=table(100_000, {"id": col(100_000)}))
    select = star_select(TableRef("orders"))
    result = optimize(select, cat)
    node = _scan_under_project(result)
    assert isinstance(node, PhysicalTableScan)


def test_predicate_on_non_indexed_column_uses_full_scan() -> None:
    cat = catalog(
        orders=table(
            100_000,
            {"id": col(100_000), "customer_id": col(100_000), "notes": col(100_000)},
            indexes=(index("idx_customer_id", "customer_id"),),
        )
    )
    select = star_select(TableRef("orders"), where=eq(ref("notes"), "hello"))
    result = optimize(select, cat)
    node = _scan_under_project(result)
    assert isinstance(node, PhysicalFilter)
    assert isinstance(node.input, PhysicalTableScan)


def test_chooses_cheaper_of_two_available_indexes() -> None:
    cat = catalog(
        orders=table(
            100_000,
            {"id": col(100_000), "customer_id": col(100_000), "status": col(3)},
            indexes=(
                index("idx_customer_id", "customer_id"),
                index("idx_status", "status"),
            ),
        )
    )
    # El predicado sólo casa con el índice de customer_id (muy selectivo).
    select = star_select(TableRef("orders"), where=eq(ref("customer_id"), 7))
    result = optimize(select, cat)
    node = _scan_under_project(result)
    assert isinstance(node, PhysicalIndexScan)
    assert node.index_name == "idx_customer_id"


def test_range_predicate_can_use_index_when_narrow() -> None:
    from cost_based_query_optimizer.models import HistogramBucket

    # Histograma de equi-altura: 10 cubos de 100k filas cada uno, cubriendo
    # 0..1_000_000 en anchos iguales, para que `amount > 999_990` sólo
    # cubra una fracción minúscula del último cubo (selectividad real y
    # narrow, no el valor por defecto sin histograma).
    histogram = tuple(
        HistogramBucket(
            lower_bound=i * 100_000,
            upper_bound=(i + 1) * 100_000,
            row_count=100_000,
            distinct_count=100_000,
        )
        for i in range(10)
    )
    cat = catalog(
        orders=table(
            1_000_000,
            {"id": col(1_000_000), "amount": col(1_000_000, histogram=histogram)},
            indexes=(index("idx_amount", "amount"),),
        )
    )
    select = star_select(TableRef("orders"), where=cmp(ref("amount"), BinaryOperator.GT, 999_990))
    result = optimize(select, cat)
    node = _scan_under_project(result)
    assert isinstance(node, PhysicalIndexScan)


def test_residual_conjunct_beyond_index_column_is_kept_as_filter() -> None:
    cat = catalog(
        orders=table(
            100_000,
            {"id": col(100_000), "customer_id": col(100_000), "status": col(3)},
            indexes=(index("idx_customer_id", "customer_id"),),
        )
    )
    from support import and_

    select = star_select(
        TableRef("orders"), where=and_(eq(ref("customer_id"), 7), eq(ref("status"), "PENDING"))
    )
    result = optimize(select, cat)
    node = _scan_under_project(result)
    assert isinstance(node, PhysicalFilter)
    assert isinstance(node.input, PhysicalIndexScan)
    assert node.input.index_name == "idx_customer_id"
