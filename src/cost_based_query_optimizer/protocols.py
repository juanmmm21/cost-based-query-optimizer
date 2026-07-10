"""Interfaces pluggables del optimizador: cómo se estima la cardinalidad de
un predicado/join (`CardinalityEstimator`) y cómo se traduce esa
cardinalidad en un coste comparable entre alternativas (`CostModel`).

Se exponen como `Protocol` — el mismo patrón que `ErrorFormatter` en
`sql-query-parser` o `DeadlockVictimPolicy` en
`lock-manager-deadlock-detector` — porque ambas decisiones son, por
naturaleza, aproximaciones estadísticas sustituibles: `nanosql` podría
inyectar un `CardinalityEstimator` que use estadísticas reales del storage
engine activo (histogramas más finos, sketches de cardinalidad) o un
`CostModel` calibrado con constantes de hardware medidas, sin tocar
`physical.py`. Las implementaciones por defecto de este módulo son
deliberadamente simples y con cada constante nombrada y justificada — un
optimizador de producción real (Postgres, SQL Server) usa fórmulas mucho
más elaboradas, pero el contrato (`Protocol`) es el mismo.
"""

from __future__ import annotations

import math
from typing import Protocol, runtime_checkable

from .models import (
    BinaryOp,
    BinaryOperator,
    ColumnRef,
    Expression,
    Literal,
    TableStatistics,
    UnaryOp,
    UnaryOperator,
)

_RANGE_OPERATORS = frozenset(
    {BinaryOperator.LT, BinaryOperator.LE, BinaryOperator.GT, BinaryOperator.GE}
)

_FLIPPED_OPERATOR = {
    BinaryOperator.LT: BinaryOperator.GT,
    BinaryOperator.LE: BinaryOperator.GE,
    BinaryOperator.GT: BinaryOperator.LT,
    BinaryOperator.GE: BinaryOperator.LE,
}


@runtime_checkable
class CardinalityEstimator(Protocol):
    """Estima cuántas filas produce un predicado de una sola tabla, o un
    join entre una tabla ya acumulada (de la que no se conserva distribución
    detallada) y la siguiente tabla base añadida por la reordenación de
    joins (ver `physical.py`, que es left-deep: el lado nuevo de cada paso
    siempre es una tabla base con estadísticas exactas)."""

    def estimate_selectivity(self, predicate: Expression, stats: TableStatistics) -> float:
        """Fracción esperada de filas de `stats` que cumplen `predicate`,
        en `[0, 1]`."""
        ...

    def estimate_join_cardinality(
        self,
        left_rows: float,
        right_rows: float,
        join_column_on_new_table: ColumnRef | None,
        new_table_stats: TableStatistics,
        has_unrecognized_predicate: bool,
    ) -> float:
        """Cardinalidad estimada de unir `left_rows` filas acumuladas con
        `right_rows` filas de la tabla base que se añade en este paso.

        - `join_column_on_new_table` es la columna del lado de la tabla
          nueva en una igualdad de join reconocida (`a.x = b.y`), ya
          reorientada por `predicates.orient_equi_condition`, o `None` si el
          conjunto de condiciones de este paso no contiene ninguna igualdad
          directa entre columnas.
        - `has_unrecognized_predicate` distingue "no hay ninguna condición
          en absoluto" (producto cartesiano real) de "hay condición pero no
          es una igualdad reconocible" (p. ej. un join por rango): en el
          segundo caso conviene reducir algo la cardinalidad aunque no se
          pueda estimar con precisión, en vez de tratarlo como cartesiano.
        """
        ...


@runtime_checkable
class CostModel(Protocol):
    """Convierte cardinalidades estimadas en un coste comparable. Todas las
    unidades son abstractas (no milisegundos reales) — sólo importa el orden
    relativo entre alternativas, exactamente igual que el optimizador de
    cualquier motor de bases de datos real."""

    def cost_table_scan(self, row_count: int) -> float: ...

    def cost_index_scan(self, matched_rows: float, table_row_count: int) -> float: ...

    def cost_filter(self, input_cost: float, input_rows: float) -> float: ...

    def cost_nested_loop_join(
        self, left_cost: float, left_rows: float, right_cost: float, output_rows: float
    ) -> float: ...

    def cost_hash_join(
        self,
        build_cost: float,
        build_rows: float,
        probe_cost: float,
        probe_rows: float,
        output_rows: float,
    ) -> float: ...

    def cost_sort(self, input_cost: float, input_rows: float) -> float: ...

    def cost_aggregate(self, input_cost: float, input_rows: float, output_rows: float) -> float: ...


class SimpleCardinalityEstimator:
    """Implementación por defecto de `CardinalityEstimator`.

    Supuestos documentados (todos constantes nombradas, ninguna cifra
    mágica dispersa en el código — ver `../CLAUDE.md`):

    - **Independencia de atributos** en conjunciones (`AND`): la
      selectividad combinada es el producto de las individuales. Es la
      misma simplificación que usa cualquier optimizador de libro de texto
      (y la mayoría de motores reales, salvo que exploten estadísticas
      multi-columna explícitas, que este módulo no modela).
    - **Inclusión-exclusión** para `OR`: `P(a∨b) = P(a)+P(b)-P(a)·P(b)`,
      asumiendo independencia igual que en `AND`.
    - **Contención unilateral** para el tamaño de un join: al no conservar
      distribución alguna del lado acumulado (posiblemente ya el resultado
      de varios joins previos), se asume que todo valor de la columna de
      join del lado nuevo (tabla base, con estadísticas exactas) tiene
      correspondencia en el lado acumulado — la misma asunción de
      "contención" que un optimizador real usa entre claves foránea/primaria.
    """

    #: Sin `distinct_count` fiable para una columna, se asume una
    #: cardinalidad de valores distintos moderada (ni única, que
    #: sobrestimaría la selectividad de una igualdad, ni un único valor
    #: global, que la subestimaría) en vez de fallar.
    DEFAULT_DISTINCT_COUNT_FOR_EQUALITY = 10

    #: Valor clásico de la literatura de optimizadores para un predicado de
    #: rango de una sola comparación sin histograma disponible.
    DEFAULT_RANGE_SELECTIVITY = 1.0 / 3.0

    #: Para cualquier forma de predicado que no sea AND/OR/NOT ni una
    #: comparación reconocida contra un literal: ni tan optimista como no
    #: filtrar nada, ni tan pesimista como asumir alta selectividad sin base
    #: para estimarla.
    DEFAULT_UNKNOWN_SELECTIVITY = 0.5

    #: Para una condición de join presente pero no reconocida como igualdad
    #: directa entre columnas (p. ej. un join por rango).
    DEFAULT_JOIN_SELECTIVITY = 0.1

    def estimate_selectivity(self, predicate: Expression, stats: TableStatistics) -> float:
        selectivity = self._selectivity(predicate, stats)
        return max(0.0, min(1.0, selectivity))

    def _selectivity(self, predicate: Expression, stats: TableStatistics) -> float:
        match predicate:
            case BinaryOp(left=left, operator=BinaryOperator.AND, right=right):
                return self._selectivity(left, stats) * self._selectivity(right, stats)
            case BinaryOp(left=left, operator=BinaryOperator.OR, right=right):
                left_sel = self._selectivity(left, stats)
                right_sel = self._selectivity(right, stats)
                return left_sel + right_sel - left_sel * right_sel
            case UnaryOp(operator=UnaryOperator.NOT, operand=operand):
                return 1.0 - self._selectivity(operand, stats)
            case BinaryOp(left=ColumnRef() as column, operator=BinaryOperator.EQ, right=Literal()):
                return self._equality_selectivity(column, stats)
            case BinaryOp(left=Literal(), operator=BinaryOperator.EQ, right=ColumnRef() as column):
                return self._equality_selectivity(column, stats)
            case BinaryOp(left=ColumnRef() as column, operator=BinaryOperator.NEQ, right=Literal()):
                return 1.0 - self._equality_selectivity(column, stats)
            case BinaryOp(left=Literal(), operator=BinaryOperator.NEQ, right=ColumnRef() as column):
                return 1.0 - self._equality_selectivity(column, stats)
            case BinaryOp(
                left=ColumnRef() as column, operator=operator, right=Literal(value=value)
            ) if operator in _RANGE_OPERATORS:
                return self._range_selectivity(column, operator, value, stats)
            case BinaryOp(
                left=Literal(value=value), operator=operator, right=ColumnRef() as column
            ) if operator in _RANGE_OPERATORS:
                return self._range_selectivity(column, _FLIPPED_OPERATOR[operator], value, stats)
            case _:
                return self.DEFAULT_UNKNOWN_SELECTIVITY

    def _equality_selectivity(self, column: ColumnRef, stats: TableStatistics) -> float:
        column_stats = stats.columns.get(column.name)
        if column_stats is not None and column_stats.distinct_count > 0:
            return 1.0 / column_stats.distinct_count
        return 1.0 / self.DEFAULT_DISTINCT_COUNT_FOR_EQUALITY

    def _range_selectivity(
        self, column: ColumnRef, operator: BinaryOperator, value: object, stats: TableStatistics
    ) -> float:
        column_stats = stats.columns.get(column.name)
        if column_stats is None or not column_stats.histogram or not isinstance(value, int | float):
            return self.DEFAULT_RANGE_SELECTIVITY

        total_rows = sum(bucket.row_count for bucket in column_stats.histogram)
        if total_rows == 0:
            return 0.0

        covered_rows = 0.0
        for bucket in column_stats.histogram:
            width = bucket.upper_bound - bucket.lower_bound
            if width <= 0:
                satisfies = self._satisfies(operator, bucket.lower_bound, value)
                covered_rows += bucket.row_count if satisfies else 0.0
                continue
            if operator in (BinaryOperator.LT, BinaryOperator.LE):
                covered_lo, covered_hi = bucket.lower_bound, min(bucket.upper_bound, value)
            else:
                covered_lo, covered_hi = max(bucket.lower_bound, value), bucket.upper_bound
            overlap = max(0.0, covered_hi - covered_lo)
            covered_rows += bucket.row_count * (overlap / width)
        return covered_rows / total_rows

    @staticmethod
    def _satisfies(operator: BinaryOperator, column_value: float, literal: float) -> bool:
        if operator is BinaryOperator.LT:
            return column_value < literal
        if operator is BinaryOperator.LE:
            return column_value <= literal
        if operator is BinaryOperator.GT:
            return column_value > literal
        return column_value >= literal

    def estimate_join_cardinality(
        self,
        left_rows: float,
        right_rows: float,
        join_column_on_new_table: ColumnRef | None,
        new_table_stats: TableStatistics,
        has_unrecognized_predicate: bool,
    ) -> float:
        if join_column_on_new_table is not None:
            column_stats = new_table_stats.columns.get(join_column_on_new_table.name)
            if column_stats is not None and column_stats.distinct_count > 0:
                distinct = float(column_stats.distinct_count)
            else:
                distinct = max(right_rows, 1.0)
            return (left_rows * right_rows) / max(distinct, 1.0)
        if has_unrecognized_predicate:
            return left_rows * right_rows * self.DEFAULT_JOIN_SELECTIVITY
        return left_rows * right_rows


class SimpleCostModel:
    """Implementación por defecto de `CostModel`, inspirada en el modelo
    clásico de coste basado en E/S + CPU (System R / Postgres): cada
    constante representa un "precio" relativo de una operación elemental,
    no un tiempo real medido — lo único que garantiza el optimizador es que
    la ordenación relativa de costes entre planes alternativos es estable.
    """

    #: Coste de leer una página en un recorrido secuencial (I/O contiguo).
    SEQ_PAGE_COST = 1.0

    #: Coste de leer una página en un recorrido por índice (I/O disperso) —
    #: más caro que el secuencial, igual que en disco/SSD real bajo
    #: concurrencia, aunque este módulo no mide hardware real.
    RANDOM_PAGE_COST = 4.0

    #: Coste de procesar una fila en CPU (evaluar una proyección, copiarla).
    CPU_TUPLE_COST = 0.01

    #: Coste de una operación de CPU más cara por fila (evaluar un
    #: predicado, calcular un hash, comparar en un sort).
    CPU_OPERATOR_COST = 0.0025

    #: Coste de procesar una entrada de índice (más barato que una fila
    #: completa de la tabla: sólo se toca la clave, no el registro entero,
    #: salvo que haga falta el residual filter aparte).
    CPU_INDEX_TUPLE_COST = 0.005

    #: Densidad de filas por página asumida **sólo** para estimar E/S — este
    #: módulo no persiste páginas reales (a diferencia de
    #: `bplus-tree-storage-engine`/`lsm-tree-engine`); es la única suposición
    #: física del modelo de coste, documentada explícitamente en vez de
    #: dispersa como número mágico.
    ROWS_PER_PAGE = 100

    def cost_table_scan(self, row_count: int) -> float:
        pages = max(1, -(-row_count // self.ROWS_PER_PAGE))  # ceil sin importar math
        return pages * self.SEQ_PAGE_COST + row_count * self.CPU_TUPLE_COST

    def cost_index_scan(self, matched_rows: float, table_row_count: int) -> float:
        if matched_rows <= 0:
            # Coste mínimo: localizar la posición en el índice aunque no
            # haya coincidencias que materializar.
            return self.RANDOM_PAGE_COST
        # A diferencia de `cost_table_scan`, aquí NO se amortiguan las filas
        # coincidentes entre páginas (`ROWS_PER_PAGE`): este índice es
        # secundario, no clúster (ver `models.IndexStatistics`), así que sus
        # filas coincidentes están dispersas por la tabla sin garantía de
        # localidad — cada una implica, en el caso pesimista estándar de
        # cualquier optimizador de libro de texto, su propia página
        # aleatoria. Es precisamente esto lo que hace que un `IndexScan`
        # deje de ganar frente a un `TableScan` a partir de cierta
        # selectividad (ver README, sección de benchmarks).
        traversal_cost = _log2(max(table_row_count, 2)) * self.CPU_OPERATOR_COST
        return (
            matched_rows * self.RANDOM_PAGE_COST
            + matched_rows * self.CPU_INDEX_TUPLE_COST
            + traversal_cost
        )

    def cost_filter(self, input_cost: float, input_rows: float) -> float:
        return input_cost + input_rows * self.CPU_OPERATOR_COST

    def cost_nested_loop_join(
        self, left_cost: float, left_rows: float, right_cost: float, output_rows: float
    ) -> float:
        # El lado derecho se recorre entero por cada fila del izquierdo (sin
        # caché) — la fórmula clásica de nested-loop join.
        return left_cost + left_rows * right_cost + output_rows * self.CPU_TUPLE_COST

    def cost_hash_join(
        self,
        build_cost: float,
        build_rows: float,
        probe_cost: float,
        probe_rows: float,
        output_rows: float,
    ) -> float:
        build_phase = build_cost + build_rows * self.CPU_OPERATOR_COST
        probe_phase = probe_cost + probe_rows * self.CPU_TUPLE_COST
        return build_phase + probe_phase + output_rows * self.CPU_TUPLE_COST

    def cost_sort(self, input_cost: float, input_rows: float) -> float:
        n = max(input_rows, 1.0)
        return input_cost + input_rows * _log2(n) * self.CPU_OPERATOR_COST

    def cost_aggregate(self, input_cost: float, input_rows: float, output_rows: float) -> float:
        return (
            input_cost + input_rows * self.CPU_OPERATOR_COST * 2 + output_rows * self.CPU_TUPLE_COST
        )


def _log2(value: float) -> float:
    return math.log2(value) if value > 1 else 0.0
