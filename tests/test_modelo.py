"""Tests de la lógica del modelo: selección de variables, división, pipeline y métricas."""
import numpy as np
import pytest

from domain.modelo import calcular_metricas, construir_pipeline, seleccionar_variables

EXCLUIR = ["RowNumber", "CustomerId", "Surname", "Complain"]


def test_seleccion_all_excluye_identificadores_y_fuga(df):
    columnas = seleccionar_variables(df, {"seleccion": "ALL", "excluir": EXCLUIR}, "Exited")
    assert len(columnas) == 13
    assert not set(EXCLUIR + ["Exited"]) & set(columnas)


def test_seleccion_numeric_solo_numericas(df):
    columnas = seleccionar_variables(df, {"seleccion": "NUMERIC", "excluir": EXCLUIR}, "Exited")
    assert "Geography" not in columnas
    assert all(df[c].dtype.kind in "biuf" for c in columnas)


def test_seleccion_custom(df):
    columnas = seleccionar_variables(df, {"seleccion": "CUSTOM", "custom": ["Age", "Geography"], "excluir": EXCLUIR}, "Exited")
    assert columnas == ["Age", "Geography"]


def test_custom_con_columna_inexistente_falla(df):
    with pytest.raises(ValueError, match="no existen"):
        seleccionar_variables(df, {"seleccion": "CUSTOM", "custom": ["NoExiste"]}, "Exited")


def test_custom_con_columna_excluida_falla(df):
    with pytest.raises(ValueError, match="excluir"):
        seleccionar_variables(df, {"seleccion": "CUSTOM", "custom": ["Complain"], "excluir": EXCLUIR}, "Exited")


def test_seleccion_desconocida_falla(df):
    with pytest.raises(ValueError, match="ALL, NUMERIC o CUSTOM"):
        seleccionar_variables(df, {"seleccion": "OTRA"}, "Exited")


def test_division_estratificada_y_reproducible(particion, df, cfg):
    entrenamiento, prueba = particion
    assert len(entrenamiento) + len(prueba) == len(df)
    assert len(prueba) == pytest.approx(len(df) * cfg["division"]["test_size"])
    assert entrenamiento["Exited"].mean() == pytest.approx(prueba["Exited"].mean(), abs=0.005)
    assert set(entrenamiento.index).isdisjoint(prueba.index)


def test_pipeline_aprende_algo(particion, cfg_features):
    entrenamiento, prueba = particion
    columnas = seleccionar_variables(entrenamiento, cfg_features, "Exited")
    modelo = construir_pipeline(entrenamiento, columnas, "logistic_regression", {"max_iter": 1000})
    modelo.fit(entrenamiento[columnas], entrenamiento["Exited"])
    metricas = calcular_metricas(prueba["Exited"], modelo.predict_proba(prueba)[:, 1], 0.5)
    assert metricas["auc"] > 0.7  # mucho mejor que el azar (0.5)


def test_tipo_de_modelo_desconocido_falla(df):
    with pytest.raises(ValueError, match="desconocido"):
        construir_pipeline(df, ["Age"], "red_neuronal", {})


def test_metricas_con_prediccion_perfecta():
    y = np.array([0, 0, 1, 1])
    metricas = calcular_metricas(y, np.array([0.1, 0.2, 0.8, 0.9]), 0.5)
    assert metricas == {"auc": 1.0, "accuracy": 1.0, "precision": 1.0, "recall": 1.0, "f1": 1.0}
