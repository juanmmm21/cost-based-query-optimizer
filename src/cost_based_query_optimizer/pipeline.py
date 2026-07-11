"""Punto de entrada público del optimizador: `optimize()` encadena
`logical.build_logical_plan` y `physical.build_physical_plan`; `explain()`
imprime el plan físico resultante en un formato legible (estilo `EXPLAIN` de
cualquier motor SQL real), usado tanto por la CLI de demostración como por
las aserciones de los tests.
"""

from __future__ import annotations

from dataclasses import dataclass

from .logical import build_logical_plan
from .models import (
    Between,
    BinaryOp,
    BinaryOperator,
    Catalog,
    ColumnRef,
    Expression,
    FunctionCall,
    InList,
    IsNull,
    Like,
    Literal,
    SelectStatement,
    Star,
    UnaryOp,
    UnaryOperator,
)
from .physical import build_physical_plan
from .plan import (
    PhysicalDistinct,
    PhysicalFilter,
    PhysicalHashAggregate,
    PhysicalHashJoin,
    PhysicalIndexScan,
    PhysicalLimit,
    PhysicalNestedLoopJoin,
    PhysicalOperator,
    PhysicalProject,
    PhysicalSort,
    PhysicalTableScan,
)
from .protocols import (
    CardinalityEstimator,
    CostModel,
    SimpleCardinalityEstimator,
    SimpleCostModel,
)


@dataclass(frozen=True, slots=True)
class OptimizationResult:
    physical_plan: PhysicalOperator

    @property
    def estimated_cost(self) -> float:
        return self.physical_plan.estimated_cost

    @property
    def estimated_rows(self) -> float:
        return self.physical_plan.estimated_rows


def optimize(
    select: SelectStatement,
    catalog: Catalog,
    *,
    cost_model: CostModel | None = None,
    cardinality_estimator: CardinalityEstimator | None = None,
) -> OptimizationResult:
    """Optimiza `select` contra `catalog`, devolviendo el plan físico de
    coste mínimo. Determinista: la misma entrada produce siempre exactamente
    el mismo plan (ver `physical.py`, que no depende en ningún punto de la
    reordenación de joins del orden de iteración de un `dict`/`set` — sólo
    de máscaras de bits sobre listas ya ordenadas)."""

    resolved_cost_model: CostModel = cost_model if cost_model is not None else SimpleCostModel()
    resolved_estimator: CardinalityEstimator = (
        cardinality_estimator if cardinality_estimator is not None else SimpleCardinalityEstimator()
    )
    logical_plan = build_logical_plan(select)
    physical_plan = build_physical_plan(
        logical_plan, catalog, resolved_cost_model, resolved_estimator
    )
    return OptimizationResult(physical_plan)


def explain(plan: PhysicalOperator) -> str:
    """Representación textual indentada del plan físico: una línea por
    operador con su cardinalidad y coste estimados, hijos indentados
    debajo — el mismo formato que la CLI (`__main__.py`) usa para `demo`."""

    lines: list[str] = []
    _explain_node(plan, 0, lines)
    return "\n".join(lines)


def _explain_node(node: PhysicalOperator, depth: int, lines: list[str]) -> None:
    lines.append("  " * depth + _describe(node))
    for child in _children(node):
        _explain_node(child, depth + 1, lines)


def _describe(node: PhysicalOperator) -> str:
    stats = f"rows={node.estimated_rows:.1f} cost={node.estimated_cost:.4f}"
    if isinstance(node, PhysicalTableScan):
        return f"TableScan({node.table.effective_name}) {stats}"
    if isinstance(node, PhysicalIndexScan):
        return (
            f"IndexScan({node.table.effective_name} via {node.index_name}, "
            f"{_format_expr(node.index_predicate)}) {stats}"
        )
    if isinstance(node, PhysicalFilter):
        return f"Filter({_format_expr(node.predicate)}) {stats}"
    if isinstance(node, PhysicalNestedLoopJoin):
        condition = _format_expr(node.condition) if node.condition is not None else "TRUE"
        return f"NestedLoopJoin[{node.join_type.name}]({condition}) {stats}"
    if isinstance(node, PhysicalHashJoin):
        return f"HashJoin[{node.join_type.name}]({_format_expr(node.condition)}) {stats}"
    if isinstance(node, PhysicalHashAggregate):
        group_desc = ", ".join(_format_expr(g) for g in node.group_by)
        agg_desc = ", ".join(_format_expr(a) for a in node.aggregates)
        return f"HashAggregate(group_by=[{group_desc}], aggregates=[{agg_desc}]) {stats}"
    if isinstance(node, PhysicalDistinct):
        return f"Distinct {stats}"
    if isinstance(node, PhysicalProject):
        items_desc = ", ".join(_format_expr(item.expression) for item in node.items)
        return f"Project([{items_desc}]) {stats}"
    if isinstance(node, PhysicalSort):
        order_desc = ", ".join(
            f"{_format_expr(item.expression)} {item.direction.name}" for item in node.order_by
        )
        return f"Sort({order_desc}) {stats}"
    if isinstance(node, PhysicalLimit):
        return f"Limit(limit={node.limit}, offset={node.offset}) {stats}"
    return f"{type(node).__name__} {stats}"


def _children(node: PhysicalOperator) -> tuple[PhysicalOperator, ...]:
    if isinstance(
        node,
        PhysicalFilter
        | PhysicalProject
        | PhysicalSort
        | PhysicalLimit
        | PhysicalHashAggregate
        | PhysicalDistinct,
    ):
        return (node.input,)
    if isinstance(node, PhysicalNestedLoopJoin | PhysicalHashJoin):
        return (node.left, node.right)
    return ()


_OPERATOR_SYMBOLS = {
    BinaryOperator.OR: "OR",
    BinaryOperator.AND: "AND",
    BinaryOperator.EQ: "=",
    BinaryOperator.NEQ: "!=",
    BinaryOperator.LT: "<",
    BinaryOperator.LE: "<=",
    BinaryOperator.GT: ">",
    BinaryOperator.GE: ">=",
    BinaryOperator.ADD: "+",
    BinaryOperator.SUB: "-",
    BinaryOperator.MUL: "*",
    BinaryOperator.DIV: "/",
    BinaryOperator.MOD: "%",
}


def _format_expr(expr: Expression) -> str:
    if isinstance(expr, Literal):
        return "NULL" if expr.value is None else repr(expr.value)
    if isinstance(expr, ColumnRef):
        return f"{expr.table}.{expr.name}" if expr.table is not None else expr.name
    if isinstance(expr, Star):
        return "*"
    if isinstance(expr, BinaryOp):
        symbol = _OPERATOR_SYMBOLS[expr.operator]
        return f"({_format_expr(expr.left)} {symbol} {_format_expr(expr.right)})"
    if isinstance(expr, UnaryOp):
        if expr.operator is UnaryOperator.NOT:
            return f"NOT {_format_expr(expr.operand)}"
        symbol = "-" if expr.operator is UnaryOperator.NEG else "+"
        return f"{symbol}{_format_expr(expr.operand)}"
    if isinstance(expr, FunctionCall):
        if expr.star_argument:
            return f"{expr.name}(*)"
        prefix = "DISTINCT " if expr.distinct else ""
        inner = ", ".join(_format_expr(a) for a in expr.arguments)
        return f"{expr.name}({prefix}{inner})"
    if isinstance(expr, Between):
        negated = "NOT " if expr.negated else ""
        return (
            f"{_format_expr(expr.operand)} {negated}BETWEEN "
            f"{_format_expr(expr.low)} AND {_format_expr(expr.high)}"
        )
    if isinstance(expr, InList):
        negated = "NOT " if expr.negated else ""
        inner = ", ".join(_format_expr(v) for v in expr.values)
        return f"{_format_expr(expr.operand)} {negated}IN ({inner})"
    if isinstance(expr, Like):
        negated = "NOT " if expr.negated else ""
        return f"{_format_expr(expr.operand)} {negated}LIKE {_format_expr(expr.pattern)}"
    if isinstance(expr, IsNull):
        negated = "NOT " if expr.negated else ""
        return f"{_format_expr(expr.operand)} IS {negated}NULL"
    return type(expr).__name__
