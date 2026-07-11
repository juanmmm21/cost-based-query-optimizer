"""Traducción del AST de entrada (`models.SelectStatement`) al plan lógico
(`plan.py`) — una correspondencia estructural directa, sin reescritura (ver
el docstring de `plan.py` para la justificación de por qué la reescritura
guiada por coste vive únicamente en `physical.py`)."""

from __future__ import annotations

from collections.abc import Iterator

from .models import (
    AGGREGATE_FUNCTION_NAMES,
    BinaryOp,
    Expression,
    FunctionCall,
    SelectStatement,
    UnaryOp,
)
from .plan import (
    LogicalAggregate,
    LogicalDistinct,
    LogicalFilter,
    LogicalJoin,
    LogicalLimit,
    LogicalOperator,
    LogicalProject,
    LogicalScan,
    LogicalSort,
)


def build_logical_plan(select: SelectStatement) -> LogicalOperator:
    """Construye el plan lógico de un `SELECT`, en el mismo orden en que se
    aplican sus cláusulas: `FROM`/`JOIN` → `WHERE` → `GROUP BY`/agregados
    (+ `HAVING`) → columnas del `SELECT` → `DISTINCT` → `ORDER BY` →
    `LIMIT`/`OFFSET`."""

    node: LogicalOperator = LogicalScan(select.from_table)
    for join in select.joins:
        node = LogicalJoin(
            left=node,
            right=LogicalScan(join.table),
            join_type=join.join_type,
            condition=join.on,
        )

    if select.where is not None:
        node = LogicalFilter(input=node, predicate=select.where)

    aggregates = _collect_aggregate_calls(select)
    if select.group_by or aggregates:
        node = LogicalAggregate(input=node, group_by=select.group_by, aggregates=aggregates)
        if select.having is not None:
            node = LogicalFilter(input=node, predicate=select.having)

    node = LogicalProject(input=node, items=select.columns)

    if select.distinct:
        node = LogicalDistinct(input=node)

    if select.order_by:
        node = LogicalSort(input=node, order_by=select.order_by)

    if select.limit is not None or select.offset is not None:
        node = LogicalLimit(input=node, limit=select.limit, offset=select.offset)

    return node


def _collect_aggregate_calls(select: SelectStatement) -> tuple[FunctionCall, ...]:
    """Todas las llamadas a función agregada (`COUNT`/`SUM`/`AVG`/`MIN`/`MAX`)
    que aparecen en las columnas proyectadas o en `HAVING`, sin duplicados y
    en orden de aparición. No se valida anidamiento inválido de agregados
    (p. ej. `SUM(COUNT(*))`) — se asume una consulta ya semánticamente
    válida, la misma responsabilidad que `sql-query-parser` deja fuera de
    su propio alcance."""

    found: dict[FunctionCall, None] = {}
    for item in select.columns:
        for call in _iter_function_calls(item.expression):
            found[call] = None
    if select.having is not None:
        for call in _iter_function_calls(select.having):
            found[call] = None
    return tuple(found)


def _iter_function_calls(expr: Expression) -> Iterator[FunctionCall]:
    match expr:
        case FunctionCall(name=name, arguments=arguments) as call:
            if name.upper() in AGGREGATE_FUNCTION_NAMES:
                yield call
            for argument in arguments:
                yield from _iter_function_calls(argument)
        case BinaryOp(left=left, right=right):
            yield from _iter_function_calls(left)
            yield from _iter_function_calls(right)
        case UnaryOp(operand=operand):
            yield from _iter_function_calls(operand)
        case _:
            return
