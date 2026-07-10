"""Tipos de entrada del optimizador: subconjunto del AST de `sql-query-parser`
redefinido de forma independiente (mismo nombre y forma de campos, cero
imports cruzados — ver `../AGENTS.md` y la regla de integración de
`../CLAUDE.md`), más las estadísticas de tabla que produce el storage engine
activo.

Alcance deliberado del AST redefinido aquí: sólo `SelectStatement` y las
expresiones/cláusulas que aparecen dentro de un `SELECT` (fuente, joins,
filtro, agregación, orden). Las sentencias DML (`INSERT`/`UPDATE`/`DELETE`) y
DDL (`CREATE TABLE`/`CREATE INDEX`) del parser real no se redefinen: un
optimizador basado en coste planifica lecturas, no escrituras — `nanosql`
enruta esas sentencias directamente al motor de ejecución sin pasar por
aquí. La única extensión razonable a futuro (no implementada, comunicada
aparte) sería reutilizar la selección de scan de este módulo para el
`WHERE` de `UPDATE`/`DELETE`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto

from .errors import MissingStatisticsError

# ---------------------------------------------------------------------------
# Expresiones (subconjunto de sql-query-parser/src/sql_query_parser/models.py)
# ---------------------------------------------------------------------------


class Expression:
    """Clase marcadora para cualquier nodo de expresión del AST de entrada."""


class LiteralType(Enum):
    INTEGER = auto()
    FLOAT = auto()
    STRING = auto()
    BOOLEAN = auto()
    NULL = auto()


LiteralValue = int | float | str | bool | None


@dataclass(frozen=True, slots=True)
class Literal(Expression):
    value: LiteralValue
    literal_type: LiteralType


@dataclass(frozen=True, slots=True)
class ColumnRef(Expression):
    """Referencia a una columna, opcionalmente cualificada por tabla/alias
    (`t.col`). Sin cualificar sólo es planificable si exactamente una tabla
    de la consulta define esa columna (ver `predicates.resolve_table_index`)."""

    name: str
    table: str | None = None


@dataclass(frozen=True, slots=True)
class Star(Expression):
    """El comodín `*`, válido como único `SelectItem` de un `SELECT *`."""


class BinaryOperator(Enum):
    OR = auto()
    AND = auto()
    EQ = auto()
    NEQ = auto()
    LT = auto()
    LE = auto()
    GT = auto()
    GE = auto()
    ADD = auto()
    SUB = auto()
    MUL = auto()
    DIV = auto()
    MOD = auto()


#: Operadores de comparación reconocidos por el estimador de cardinalidad
#: (ver `protocols.py`). El resto de `BinaryOperator` (aritméticos) sólo
#: aparecen dentro de proyecciones/expresiones de valor, nunca como forma de
#: un predicado booleano completo, así que no necesitan selectividad propia.
COMPARISON_OPERATORS = frozenset(
    {
        BinaryOperator.EQ,
        BinaryOperator.NEQ,
        BinaryOperator.LT,
        BinaryOperator.LE,
        BinaryOperator.GT,
        BinaryOperator.GE,
    }
)


@dataclass(frozen=True, slots=True)
class BinaryOp(Expression):
    left: Expression
    operator: BinaryOperator
    right: Expression


class UnaryOperator(Enum):
    NEG = auto()
    POS = auto()
    NOT = auto()


@dataclass(frozen=True, slots=True)
class UnaryOp(Expression):
    operator: UnaryOperator
    operand: Expression


@dataclass(frozen=True, slots=True)
class Between(Expression):
    operand: Expression
    low: Expression
    high: Expression
    negated: bool = False


@dataclass(frozen=True, slots=True)
class InList(Expression):
    operand: Expression
    values: tuple[Expression, ...]
    negated: bool = False


@dataclass(frozen=True, slots=True)
class Like(Expression):
    operand: Expression
    pattern: Expression
    negated: bool = False


@dataclass(frozen=True, slots=True)
class IsNull(Expression):
    operand: Expression
    negated: bool = False


@dataclass(frozen=True, slots=True)
class FunctionCall(Expression):
    """Llamada a función. El optimizador sólo interpreta semánticamente las
    cinco agregadas clásicas (`AGGREGATE_FUNCTION_NAMES`) para decidir si un
    `SELECT` necesita un operador de agregación; cualquier otro nombre se
    trata como función escalar opaca de sólo-proyección."""

    name: str
    arguments: tuple[Expression, ...] = ()
    distinct: bool = False
    star_argument: bool = False


AGGREGATE_FUNCTION_NAMES = frozenset({"COUNT", "SUM", "AVG", "MIN", "MAX"})


# ---------------------------------------------------------------------------
# Cláusulas y fragmentos de un SELECT
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SelectItem:
    expression: Expression
    alias: str | None = None


@dataclass(frozen=True, slots=True)
class TableRef:
    name: str
    alias: str | None = None

    @property
    def effective_name(self) -> str:
        """Nombre por el que se referencia esta tabla dentro de la propia
        consulta (en `ColumnRef.table` y en el resto de la sentencia): el
        alias si existe, si no el nombre real."""
        return self.alias if self.alias is not None else self.name


class JoinType(Enum):
    INNER = auto()
    LEFT = auto()
    RIGHT = auto()
    FULL = auto()


@dataclass(frozen=True, slots=True)
class Join:
    join_type: JoinType
    table: TableRef
    on: Expression


class OrderDirection(Enum):
    ASC = auto()
    DESC = auto()


@dataclass(frozen=True, slots=True)
class OrderByItem:
    expression: Expression
    direction: OrderDirection = OrderDirection.ASC


class Statement:
    """Clase marcadora para cualquier nodo de sentencia de nivel superior."""


@dataclass(frozen=True, slots=True)
class SelectStatement(Statement):
    columns: tuple[SelectItem, ...]
    from_table: TableRef
    distinct: bool = False
    joins: tuple[Join, ...] = ()
    where: Expression | None = None
    group_by: tuple[Expression, ...] = ()
    having: Expression | None = None
    order_by: tuple[OrderByItem, ...] = ()
    limit: int | None = None
    offset: int | None = None


# ---------------------------------------------------------------------------
# Estadísticas de tabla (entrada producida por el storage engine activo)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class HistogramBucket:
    """Cubo de un histograma de equi-altura sobre una columna numérica: cada
    cubo de una misma columna cubre, aproximadamente, el mismo número de
    filas — por eso basta usar sus límites para repartir selectividad de
    forma proporcional al ancho del rango consultado dentro del cubo, en vez
    de asumir uniformidad sobre el rango completo de la columna."""

    lower_bound: float
    upper_bound: float
    row_count: int
    distinct_count: int


@dataclass(frozen=True, slots=True)
class ColumnStatistics:
    """Estadísticas de una columna. `histogram` sólo se rellena para columnas
    numéricas: la distribución de valores de columnas de texto se modela
    sólo mediante `distinct_count` (ver `protocols.py`,
    `DEFAULT_RANGE_SELECTIVITY` para el resto de casos sin histograma)."""

    distinct_count: int
    null_count: int = 0
    histogram: tuple[HistogramBucket, ...] = ()


@dataclass(frozen=True, slots=True)
class IndexStatistics:
    """Un índice secundario disponible sobre la tabla. Este optimizador sólo
    explota la columna líder (`columns[0]`) de cada índice para decidir
    acceso por índice — igual que un B+Tree compuesto real sólo permite
    búsqueda eficiente por prefijo de sus columnas indexadas."""

    name: str
    columns: tuple[str, ...]
    unique: bool = False


@dataclass(frozen=True, slots=True)
class TableStatistics:
    row_count: int
    columns: dict[str, ColumnStatistics] = field(default_factory=dict)
    indexes: tuple[IndexStatistics, ...] = ()


@dataclass(frozen=True, slots=True)
class Catalog:
    """Estadísticas de todas las tablas visibles para una consulta. Envuelve
    un `dict` en vez de exponerlo directamente para que la resolución de una
    tabla ausente falle con `MissingStatisticsError` (tipada) en el punto de
    uso, en vez de un `KeyError` genérico — ver `../CLAUDE.md`, manejo de
    errores explícito."""

    tables: dict[str, TableStatistics] = field(default_factory=dict)

    def get(self, table_name: str) -> TableStatistics:
        try:
            return self.tables[table_name]
        except KeyError as exc:
            raise MissingStatisticsError(table_name) from exc
