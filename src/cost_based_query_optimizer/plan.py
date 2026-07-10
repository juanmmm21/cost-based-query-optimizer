"""Árboles de plan lógico y plan físico.

El plan lógico es una traducción **estructural** y directa del AST de
entrada (`logical.build_logical_plan`): ningún predicado se mueve, ningún
join se reordena todavía. Toda la reescritura guiada por coste — empujar
filtros hacia el scan que corresponda, reordenar joins, elegir índice vs.
full scan, elegir nested-loop vs. hash join — ocurre en un único paso al
construir el plan físico (`physical.build_physical_plan`), no como una fase
de reescritura lógica intermedia separada. Es una simplificación deliberada
frente a un optimizador de libro de texto con fases de reescritura lógica y
física independientes: para el subconjunto de SQL de este ecosistema
(sin subconsultas, sin vistas) no aporta nada partir el trabajo en dos
pasadas del árbol en vez de una — ver `../CLAUDE.md`, simplicidad sobre
complejidad.

Igual que en `models.py` (y en el propio `sql-query-parser`), cada jerarquía
usa una clase marcadora sin campos (`LogicalOperator` / `PhysicalOperator`)
en vez de una base con herencia de campos: así cada nodo declara sus propios
campos sin arrastrar el orden posicional de una superclase `dataclass`.
"""

from __future__ import annotations

from dataclasses import dataclass

from .models import Expression, FunctionCall, JoinType, OrderByItem, SelectItem, TableRef

# ---------------------------------------------------------------------------
# Plan lógico
# ---------------------------------------------------------------------------


class LogicalOperator:
    """Clase marcadora para cualquier nodo del plan lógico."""


@dataclass(frozen=True, slots=True)
class LogicalScan(LogicalOperator):
    table: TableRef


@dataclass(frozen=True, slots=True)
class LogicalFilter(LogicalOperator):
    input: LogicalOperator
    predicate: Expression


@dataclass(frozen=True, slots=True)
class LogicalJoin(LogicalOperator):
    left: LogicalOperator
    right: LogicalOperator
    join_type: JoinType
    condition: Expression


@dataclass(frozen=True, slots=True)
class LogicalAggregate(LogicalOperator):
    input: LogicalOperator
    group_by: tuple[Expression, ...]
    aggregates: tuple[FunctionCall, ...]


@dataclass(frozen=True, slots=True)
class LogicalProject(LogicalOperator):
    input: LogicalOperator
    items: tuple[SelectItem, ...]


@dataclass(frozen=True, slots=True)
class LogicalSort(LogicalOperator):
    input: LogicalOperator
    order_by: tuple[OrderByItem, ...]


@dataclass(frozen=True, slots=True)
class LogicalLimit(LogicalOperator):
    input: LogicalOperator
    limit: int | None
    offset: int | None


# ---------------------------------------------------------------------------
# Plan físico
# ---------------------------------------------------------------------------
#
# Convención compartida por todos los nodos: `estimated_rows` es la
# cardinalidad de salida estimada de ESE operador (no de su entrada) y
# `estimated_cost` es el coste acumulado de ejecutar el subárbol completo
# hasta ahí (no sólo el coste incremental de este nodo) — así que
# `plan.estimated_cost` en la raíz del árbol físico es, por construcción,
# el coste total estimado de la consulta completa: el número que el
# optimizador minimiza.
#
# `condition`/`filter`/`predicate` en los nodos de join y scan sólo llevan
# la parte del predicado que ESE operador concreto sabe aplicar (la clave de
# igualdad para `PhysicalHashJoin`, el predicado indexado para
# `PhysicalIndexScan`); cualquier resto del predicado original que no encaja
# ahí se aplica con un `PhysicalFilter` apilado encima — un único mecanismo
# de "resto" reutilizado en todos los operadores, en vez de que cada uno
# tenga su propio campo de "predicado residual".


class PhysicalOperator:
    """Clase marcadora para cualquier nodo del plan físico.

    A diferencia de `LogicalOperator` (marcador sin campos: siempre se
    recorre por `isinstance`), aquí se declaran `estimated_rows`/
    `estimated_cost` como anotaciones de clase sin dataclass propio: cada
    subclase concreta las redeclara como campos reales de su propio
    `@dataclass`, pero `physical.py`/`pipeline.py` necesitan leerlas de
    forma genérica (acumular coste subiendo por el árbol) sin tener que
    hacer `isinstance` contra las diez subclases en cada sitio. `__slots__
    = ()` en la base evita que esa declaración le añada un `__dict__` a las
    subclases con `slots=True`."""

    __slots__ = ()
    estimated_rows: float
    estimated_cost: float


@dataclass(frozen=True, slots=True)
class PhysicalTableScan(PhysicalOperator):
    table: TableRef
    estimated_rows: float
    estimated_cost: float


@dataclass(frozen=True, slots=True)
class PhysicalIndexScan(PhysicalOperator):
    """Recorrido por el índice `index_name` de `table`, aplicando
    `index_predicate` (la porción del `WHERE` que el índice puede resolver
    sobre su columna líder — ver `models.IndexStatistics`)."""

    table: TableRef
    index_name: str
    index_predicate: Expression
    estimated_rows: float
    estimated_cost: float


@dataclass(frozen=True, slots=True)
class PhysicalFilter(PhysicalOperator):
    input: PhysicalOperator
    predicate: Expression
    estimated_rows: float
    estimated_cost: float


@dataclass(frozen=True, slots=True)
class PhysicalNestedLoopJoin(PhysicalOperator):
    """Aplicable a cualquier condición de join (o a ninguna: `condition=None`
    representa un producto cartesiano) — es el único método físico sin
    restricciones, y por eso el fallback cuando no hay una igualdad que
    aprovechar."""

    left: PhysicalOperator
    right: PhysicalOperator
    join_type: JoinType
    condition: Expression | None
    estimated_rows: float
    estimated_cost: float


@dataclass(frozen=True, slots=True)
class PhysicalHashJoin(PhysicalOperator):
    """Sólo válido cuando `condition` es una igualdad entre una columna de
    cada lado (ver `physical._as_equi_condition`); `left` es el lado de
    construcción de la tabla hash, `right` el de sondeo."""

    left: PhysicalOperator
    right: PhysicalOperator
    join_type: JoinType
    condition: Expression
    estimated_rows: float
    estimated_cost: float


@dataclass(frozen=True, slots=True)
class PhysicalHashAggregate(PhysicalOperator):
    input: PhysicalOperator
    group_by: tuple[Expression, ...]
    aggregates: tuple[FunctionCall, ...]
    estimated_rows: float
    estimated_cost: float


@dataclass(frozen=True, slots=True)
class PhysicalProject(PhysicalOperator):
    input: PhysicalOperator
    items: tuple[SelectItem, ...]
    estimated_rows: float
    estimated_cost: float


@dataclass(frozen=True, slots=True)
class PhysicalSort(PhysicalOperator):
    input: PhysicalOperator
    order_by: tuple[OrderByItem, ...]
    estimated_rows: float
    estimated_cost: float


@dataclass(frozen=True, slots=True)
class PhysicalLimit(PhysicalOperator):
    input: PhysicalOperator
    limit: int | None
    offset: int | None
    estimated_rows: float
    estimated_cost: float
