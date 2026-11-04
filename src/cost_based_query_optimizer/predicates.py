"""Utilidades para descomponer y recomponer predicados booleanos, y para
resolver a qué tabla de la consulta pertenece cada `ColumnRef`.

Estas funciones son el mecanismo compartido que usa `physical.py` tanto
para empujar filtros de una sola tabla hacia su scan como para decidir en
qué paso de la reordenación de joins se aplica cada condición que involucra
a varias — ver el docstring de `physical.build_physical_plan` para el
razonamiento completo de por qué un único mecanismo basta para ambos casos.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence

from .errors import UnresolvedColumnError
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
    TableRef,
    UnaryOp,
)


def split_conjuncts(predicate: Expression | None) -> tuple[Expression, ...]:
    """Descompone una expresión unida por `AND` en la lista de sus conjuntos
    de nivel superior. `OR`/comparaciones/cualquier otra forma se devuelven
    como un único conjunto sin partir — sólo `AND` es distributivo sobre "se
    puede evaluar de forma independiente"."""

    if predicate is None:
        return ()
    if isinstance(predicate, BinaryOp) and predicate.operator is BinaryOperator.AND:
        return split_conjuncts(predicate.left) + split_conjuncts(predicate.right)
    return (predicate,)


def combine_conjuncts(conjuncts: Sequence[Expression]) -> Expression | None:
    """Inversa de `split_conjuncts`: reconstruye una única expresión `AND`
    a partir de una lista de conjuntos, en el mismo orden en que se dan
    (determinista). Devuelve `None` para una lista vacía, y el propio
    elemento (sin envolver en `AND`) para una lista de un único conjunto."""

    if not conjuncts:
        return None
    combined = conjuncts[0]
    for extra in conjuncts[1:]:
        combined = BinaryOp(left=combined, operator=BinaryOperator.AND, right=extra)
    return combined


def iter_column_refs(expr: Expression) -> Iterator[ColumnRef]:
    """Recorre recursivamente cualquier forma de expresión del subconjunto
    soportado (ver `models.py`) y produce cada `ColumnRef` que contiene."""

    match expr:
        case ColumnRef():
            yield expr
        case BinaryOp(left=left, right=right):
            yield from iter_column_refs(left)
            yield from iter_column_refs(right)
        case UnaryOp(operand=operand):
            yield from iter_column_refs(operand)
        case FunctionCall(arguments=arguments):
            for argument in arguments:
                yield from iter_column_refs(argument)
        case Between(operand=operand, low=low, high=high):
            yield from iter_column_refs(operand)
            yield from iter_column_refs(low)
            yield from iter_column_refs(high)
        case InList(operand=operand, values=values):
            yield from iter_column_refs(operand)
            for value in values:
                yield from iter_column_refs(value)
        case Like(operand=operand, pattern=pattern):
            yield from iter_column_refs(operand)
            yield from iter_column_refs(pattern)
        case IsNull(operand=operand):
            yield from iter_column_refs(operand)
        case _:
            # Literal, Star: no contienen columnas. Antes Between/InList/
            # Like/IsNull también caían aquí: un `t.a IN (1, 2)` de un join
            # no "tocaba" ninguna tabla, nunca encontraba paso de join en el
            # que aplicarse y se descartaba en silencio (resultados con
            # filas de más). Todas las formas de expresión deben recorrerse.
            return


def resolve_table_index(column: ColumnRef, tables: Sequence[TableRef], catalog: Catalog) -> int:
    """Devuelve el índice (en `tables`, en el orden de la consulta) de la
    única tabla a la que pertenece `column`.

    Si `column.table` está cualificado, debe coincidir con el alias (o, en
    su ausencia, el nombre) de exactamente una tabla de `tables`. Si no está
    cualificado, debe existir en las estadísticas de exactamente una de las
    tablas de la consulta — dos tablas con una columna del mismo nombre sin
    cualificar es una referencia ambigua, igual que en SQL estándar."""

    if column.table is not None:
        matches = [i for i, t in enumerate(tables) if t.effective_name == column.table]
        if len(matches) == 1:
            return matches[0]
        raise UnresolvedColumnError(f"{column.table}.{column.name}")

    candidates = [i for i, t in enumerate(tables) if column.name in catalog.get(t.name).columns]
    if len(candidates) == 1:
        return candidates[0]
    raise UnresolvedColumnError(column.name)


def touched_tables(
    expr: Expression, tables: Sequence[TableRef], catalog: Catalog
) -> frozenset[int]:
    """Conjunto de índices de tabla que referencia `expr`, vía
    `resolve_table_index` sobre cada `ColumnRef` que contiene."""

    return frozenset(
        resolve_table_index(column, tables, catalog) for column in iter_column_refs(expr)
    )


def as_equi_columns(expr: Expression) -> tuple[ColumnRef, ColumnRef] | None:
    """Si `expr` es una igualdad directa entre dos columnas (`a.x = b.y`),
    devuelve el par `(izquierda, derecha)` tal como aparece escrito. Ninguna
    otra forma (comparación con literal, `AND`/`OR`, rango...) es
    "equi-joinable" en el sentido que necesita `PhysicalHashJoin`."""

    if (
        isinstance(expr, BinaryOp)
        and expr.operator is BinaryOperator.EQ
        and isinstance(expr.left, ColumnRef)
        and isinstance(expr.right, ColumnRef)
    ):
        return expr.left, expr.right
    return None


def orient_equi_condition(
    expr: Expression,
    left_table_indices: frozenset[int],
    tables: Sequence[TableRef],
    catalog: Catalog,
) -> tuple[ColumnRef, ColumnRef] | None:
    """Como `as_equi_columns`, pero además reorienta el par para que el
    primer elemento sea siempre la columna del lado "izquierdo" lógico
    (`left_table_indices`) — necesario porque el `ON`/conjunto de `WHERE`
    puede escribirse en cualquier orden (`a.x = b.y` o `b.y = a.x`) sin que
    eso deba importar a la hora de construir un `PhysicalHashJoin`."""

    columns = as_equi_columns(expr)
    if columns is None:
        return None
    first, second = columns
    first_index = resolve_table_index(first, tables, catalog)
    second_index = resolve_table_index(second, tables, catalog)
    if first_index in left_table_indices and second_index not in left_table_indices:
        return first, second
    if second_index in left_table_indices and first_index not in left_table_indices:
        return second, first
    return None
