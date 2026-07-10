"""Constructores compartidos por los tests: no es un módulo `test_*.py`
(pytest no lo recolecta), sólo azúcar para construir `Catalog`/
`SelectStatement`/expresiones sin repetir boilerplate en cada archivo."""

from __future__ import annotations

from cost_based_query_optimizer.models import (
    BinaryOp,
    BinaryOperator,
    Catalog,
    ColumnRef,
    ColumnStatistics,
    HistogramBucket,
    IndexStatistics,
    Join,
    JoinType,
    Literal,
    LiteralType,
    LiteralValue,
    SelectItem,
    SelectStatement,
    Star,
    TableRef,
    TableStatistics,
)


def col(
    distinct_count: int, null_count: int = 0, histogram: tuple[HistogramBucket, ...] = ()
) -> ColumnStatistics:
    return ColumnStatistics(
        distinct_count=distinct_count, null_count=null_count, histogram=histogram
    )


def table(
    row_count: int,
    columns: dict[str, ColumnStatistics] | None = None,
    indexes: tuple[IndexStatistics, ...] = (),
) -> TableStatistics:
    return TableStatistics(row_count=row_count, columns=columns or {}, indexes=indexes)


def catalog(**tables: TableStatistics) -> Catalog:
    return Catalog(tables=dict(tables))


def index(name: str, *columns: str, unique: bool = False) -> IndexStatistics:
    return IndexStatistics(name=name, columns=tuple(columns), unique=unique)


def ref(name: str, table_name: str | None = None) -> ColumnRef:
    return ColumnRef(name=name, table=table_name)


def lit(value: LiteralValue) -> Literal:
    if isinstance(value, bool):
        return Literal(value, LiteralType.BOOLEAN)
    if isinstance(value, int):
        return Literal(value, LiteralType.INTEGER)
    if isinstance(value, float):
        return Literal(value, LiteralType.FLOAT)
    if isinstance(value, str):
        return Literal(value, LiteralType.STRING)
    return Literal(None, LiteralType.NULL)


def cmp(column: ColumnRef, operator: BinaryOperator, value: LiteralValue) -> BinaryOp:
    return BinaryOp(left=column, operator=operator, right=lit(value))


def eq(column: ColumnRef, value: LiteralValue) -> BinaryOp:
    return cmp(column, BinaryOperator.EQ, value)


def eq_columns(left: ColumnRef, right: ColumnRef) -> BinaryOp:
    return BinaryOp(left=left, operator=BinaryOperator.EQ, right=right)


def and_(*conjuncts: BinaryOp) -> BinaryOp:
    combined = conjuncts[0]
    for extra in conjuncts[1:]:
        combined = BinaryOp(left=combined, operator=BinaryOperator.AND, right=extra)
    return combined


def star_select(
    from_table: TableRef,
    joins: tuple[Join, ...] = (),
    where: BinaryOp | None = None,
    group_by: tuple[ColumnRef, ...] = (),
    order_by: tuple = (),
    limit: int | None = None,
    offset: int | None = None,
) -> SelectStatement:
    return SelectStatement(
        columns=(SelectItem(Star()),),
        from_table=from_table,
        joins=joins,
        where=where,
        group_by=group_by,
        order_by=order_by,
        limit=limit,
        offset=offset,
    )


def inner_join(table_ref: TableRef, on: BinaryOp) -> Join:
    return Join(join_type=JoinType.INNER, table=table_ref, on=on)


def outer_join(join_type: JoinType, table_ref: TableRef, on: BinaryOp) -> Join:
    return Join(join_type=join_type, table=table_ref, on=on)
