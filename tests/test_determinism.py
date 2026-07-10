"""Regla B de `../CLAUDE.md`: mismo AST + mismas estadísticas -> siempre el
mismo plan. `physical.py` no depende en ningún punto de orden de iteración
de `dict`/`set` (sólo de máscaras de bits sobre listas ya ordenadas) —
estos tests son la red de seguridad que lo comprueba en negro, sin asumir
nada sobre la implementación."""

from support import catalog, col, eq_columns, inner_join, ref, star_select, table

from cost_based_query_optimizer.models import TableRef
from cost_based_query_optimizer.pipeline import optimize


def _six_table_scenario():
    names = [f"t{i}" for i in range(6)]
    tables = {
        name: table(100 * (index + 1), {"id": col(100 * (index + 1)), "prev_id": col(50)})
        for index, name in enumerate(names)
    }
    cat = catalog(**tables)
    refs = [TableRef(name, name) for name in names]
    joins = tuple(
        inner_join(refs[i], eq_columns(ref("prev_id", refs[i].alias), ref("id", refs[i - 1].alias)))
        for i in range(1, len(refs))
    )
    select = star_select(refs[0], joins=joins)
    return select, cat


def test_repeated_optimize_calls_produce_identical_plans() -> None:
    select, cat = _six_table_scenario()
    results = [optimize(select, cat).physical_plan for _ in range(10)]
    first = results[0]
    for other in results[1:]:
        assert other == first


def test_plan_is_independent_of_catalog_dict_insertion_order() -> None:
    select, cat_a = _six_table_scenario()
    # Reconstruye el catálogo con las tablas insertadas en orden inverso: el
    # `dict` resultante compara igual (`Catalog` es un dataclass estructural)
    # pero su orden de iteración interno es distinto.
    reversed_tables = dict(reversed(list(cat_a.tables.items())))
    cat_b = type(cat_a)(tables=reversed_tables)

    plan_a = optimize(select, cat_a).physical_plan
    plan_b = optimize(select, cat_b).physical_plan
    assert plan_a == plan_b


def test_plan_is_independent_of_join_type_enum_identity_across_runs() -> None:
    # No hay estado de proceso compartido entre optimizaciones: cada llamada
    # reconstruye su propio `dict`/`_Conjunct` internos. Verifica que dos
    # sentencias estructuralmente idénticas pero construidas por separado
    # (objetos distintos) producen exactamente el mismo plan.
    select_a, cat = _six_table_scenario()
    select_b, _ = _six_table_scenario()
    assert select_a == select_b
    assert optimize(select_a, cat).physical_plan == optimize(select_b, cat).physical_plan
