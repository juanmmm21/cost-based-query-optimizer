"""Conjuntos constantes (sin columnas) y claves de GROUP BY repetidas.

Regresiones encontradas por el run de conformidad sqllogictest de nanosql:
`a JOIN b ON 1 = 0` devolvía el producto cartesiano completo porque un
conjunto que no toca ninguna tabla nunca encontraba un paso de la
reordenación en el que aplicarse; `GROUP BY a, b, b` duplicaba la columna
`b` en el esquema del agregado."""

from support import catalog, col, eq_columns, inner_join, lit, ref, star_select, table

from cost_based_query_optimizer.logical import build_logical_plan
from cost_based_query_optimizer.models import (
    BinaryOp,
    BinaryOperator,
    IsNull,
    TableRef,
)
from cost_based_query_optimizer.pipeline import optimize
from cost_based_query_optimizer.plan import (
    LogicalAggregate,
    PhysicalFilter,
    PhysicalOperator,
)

CATALOG = catalog(
    a=table(100, {"x": col(100), "y": col(10)}),
    b=table(10, {"z": col(10)}),
    c=table(1000, {"x": col(1000)}),
)


def _contains_predicate(plan: PhysicalOperator, predicate: object) -> bool:
    stack = [plan]
    while stack:
        node = stack.pop()
        if isinstance(node, PhysicalFilter) and predicate in _conjuncts(node.predicate):
            return True
        for attr in ("input", "left", "right"):
            child = getattr(node, attr, None)
            if child is not None:
                stack.append(child)
    return False


def _conjuncts(expr: object) -> list[object]:
    if isinstance(expr, BinaryOp) and expr.operator is BinaryOperator.AND:
        return _conjuncts(expr.left) + _conjuncts(expr.right)
    return [expr]


def test_constant_false_join_condition_is_kept() -> None:
    false = BinaryOp(lit(1), BinaryOperator.EQ, lit(0))
    select = star_select(TableRef("a"), joins=(inner_join(TableRef("b"), false),))
    plan = optimize(select, CATALOG).physical_plan
    assert _contains_predicate(plan, false)


def test_constant_conjunct_mixed_with_equi_join_is_kept() -> None:
    null_check = IsNull(lit(None), negated=True)
    condition = BinaryOp(eq_columns(ref("x", "a"), ref("x", "c")), BinaryOperator.AND, null_check)
    select = star_select(
        TableRef("a"),
        joins=(inner_join(TableRef("c"), condition), inner_join(TableRef("b"), lit(True))),
    )
    plan = optimize(select, CATALOG).physical_plan
    assert _contains_predicate(plan, null_check)


def test_repeated_group_by_keys_are_deduplicated() -> None:
    select = star_select(TableRef("a"), group_by=(ref("y"), ref("x"), ref("y")))
    node = build_logical_plan(select)
    while not isinstance(node, LogicalAggregate):
        node = node.input  # type: ignore[attr-defined]
    assert node.group_by == (ref("y"), ref("x"))
