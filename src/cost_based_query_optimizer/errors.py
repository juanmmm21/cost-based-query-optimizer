"""Jerarquía de excepciones tipadas del optimizador.

Todas heredan de `OptimizerError` para que un consumidor (p. ej. `nanosql`)
pueda capturar cualquier fallo del optimizador con un único `except`, pero
cada subclase es lo bastante específica como para decidir una recuperación
distinta (estadísticas ausentes vs. columna ambigua vs. sentencia no
soportada).
"""

from __future__ import annotations


class OptimizerError(Exception):
    """Base de toda excepción propia de este módulo."""


class MissingStatisticsError(OptimizerError):
    """El catálogo no tiene estadísticas para una tabla referenciada por la consulta.

    El optimizador nunca asume un `row_count` por defecto ante ausencia de
    estadísticas: estimar coste sin datos reales produciría un plan
    arbitrario y silenciosamente incorrecto, así que se prefiere fallar de
    forma explícita (ver la regla de manejo de errores explícito de
    `../CLAUDE.md`) antes que inventar una cardinalidad.
    """

    def __init__(self, table_name: str) -> None:
        self.table_name = table_name
        super().__init__(f"no hay estadísticas en el catálogo para la tabla '{table_name}'")


class UnresolvedColumnError(OptimizerError):
    """Una `ColumnRef` no se pudo resolver de forma única a una de las tablas
    de la consulta: o ninguna tabla la define (probable error de la propia
    consulta), o más de una la define y la referencia no está cualificada
    (`t.columna`) para desambiguar — exactamente el mismo caso que SQL
    estándar rechaza en tiempo de análisis semántico."""

    def __init__(self, column_name: str) -> None:
        self.column_name = column_name
        super().__init__(f"no se pudo resolver la columna '{column_name}' a una única tabla")


class UnsupportedQueryError(OptimizerError):
    """La sentencia de entrada usa una construcción fuera del subconjunto que
    este optimizador sabe planificar (ver README, sección de alcance:
    sólo `SelectStatement`, y sin agregados anidados en expresiones
    arbitrarias)."""
