"""Tests de la emulación y de la detección de drift. Sin MLflow ni Prefect."""
import numpy as np
import pandas as pd
import pytest

from domain.drift import (
    calcular_drift_features,
    columnas_afectadas,
    drift_evento,
    emular_drift,
    hay_alerta,
    resumir_reporte,
    tomar_lote,
    validar_etiquetas,
    validar_nivel,
)
from domain.modelo import seleccionar_variables


def lote_con_drift(particion, cfg, nivel, semilla=42):
    entrenamiento, prueba = particion
    rng = np.random.default_rng(semilla)
    d = cfg["drift"]
    lote = tomar_lote(prueba, "Exited", cfg["inferencia"]["tamano_lote"], nivel, d["subida_tasa_max"], rng)
    return emular_drift(lote, entrenamiento, nivel, d, rng)


# --- Emulación ---

@pytest.mark.parametrize("nivel", [-0.1, 1.5])
def test_nivel_fuera_de_rango_falla(nivel):
    with pytest.raises(ValueError, match="entre 0.0 y 1.0"):
        validar_nivel(nivel)


def test_nivel_cero_no_desplaza_variables(particion, cfg, rng):
    entrenamiento, prueba = particion
    lote = tomar_lote(prueba, "Exited", 300, 0.0, 0.3, rng)
    pd.testing.assert_frame_equal(emular_drift(lote, entrenamiento, 0.0, cfg["drift"], rng), lote)


@pytest.mark.parametrize("nivel", [0.0, 0.3, 0.6])
def test_tasa_del_target_sube_con_el_nivel(particion, cfg, rng, nivel):
    _, prueba = particion
    lote = tomar_lote(prueba, "Exited", 300, nivel, cfg["drift"]["subida_tasa_max"], rng)
    esperada = prueba["Exited"].mean() + nivel * cfg["drift"]["subida_tasa_max"]
    assert len(lote) == 300
    assert lote["Exited"].mean() == pytest.approx(esperada, abs=0.01)


def test_lote_sale_solo_de_la_parte_de_prueba(particion, rng):
    entrenamiento, prueba = particion
    lote = tomar_lote(prueba, "Exited", 300, 0.6, 0.3, rng)
    assert set(lote["CustomerId"]) <= set(prueba["CustomerId"])
    assert set(lote["CustomerId"]).isdisjoint(entrenamiento["CustomerId"])


@pytest.mark.parametrize("nivel,esperadas", [(0.0, 0), (0.3, 2), (0.6, 5), (1.0, 8)])
def test_numero_de_columnas_afectadas(cfg, nivel, esperadas):
    assert len(columnas_afectadas(nivel, cfg["drift"]["candidatas"])) == esperadas


def test_drift_desplaza_solo_las_columnas_afectadas(particion, cfg, rng):
    entrenamiento, prueba = particion
    lote = tomar_lote(prueba, "Exited", 300, 0.6, 0.3, rng)
    con_drift = emular_drift(lote, entrenamiento, 0.6, cfg["drift"], rng)
    afectadas = columnas_afectadas(0.6, cfg["drift"]["candidatas"])
    cambiadas = [c for c in lote.columns if not lote[c].equals(con_drift[c])]
    assert sorted(cambiadas) == sorted(afectadas)
    assert con_drift["Age"].mean() > lote["Age"].mean()
    assert (con_drift["Geography"] == "Germany").mean() > (lote["Geography"] == "Germany").mean()


def test_drift_conserva_enteros_y_no_modifica_el_original(particion, cfg, rng):
    entrenamiento, prueba = particion
    lote = tomar_lote(prueba, "Exited", 300, 1.0, 0.3, rng)
    copia = lote.copy()
    con_drift = emular_drift(lote, entrenamiento, 1.0, cfg["drift"], rng)
    assert con_drift["Age"].dtype == lote["Age"].dtype
    pd.testing.assert_frame_equal(lote, copia)


def test_misma_semilla_mismo_lote(particion, cfg):
    pd.testing.assert_frame_equal(lote_con_drift(particion, cfg, 0.6), lote_con_drift(particion, cfg, 0.6))


# --- Drift de evento ---

def test_lote_sin_columna_objetivo_se_rechaza():
    with pytest.raises(ValueError, match="etiquetas reales"):
        validar_etiquetas(pd.DataFrame({"Age": [30, 40]}), "Exited")


def test_lote_con_etiquetas_pendientes_se_rechaza():
    with pytest.raises(ValueError, match="sin etiqueta"):
        validar_etiquetas(pd.DataFrame({"Exited": [1, None, 0]}), "Exited")


def test_drift_evento_detecta_cambio_de_tasa():
    referencia = pd.Series([1] * 200 + [0] * 800)
    lote = pd.Series([1] * 120 + [0] * 180)  # 40 % frente a 20 %
    resultado = drift_evento(referencia, lote, alfa=0.05)
    assert resultado["drift"]
    assert resultado["diferencia"] == pytest.approx(0.2)


def test_drift_evento_sin_cambio_de_tasa():
    referencia = pd.Series([1] * 200 + [0] * 800)
    lote = pd.Series([1] * 60 + [0] * 240)  # 20 % igual que la referencia
    assert not drift_evento(referencia, lote, alfa=0.05)["drift"]


# --- Drift de features ---

def test_resumir_reporte_lee_la_fraccion():
    reporte = {"metrics": [
        {"metric_name": "ValueDrift(column=Age)", "value": 0.01},
        {"metric_name": "DriftedColumnsCount(drift_share=0.5)", "value": {"count": 5.0, "share": 0.3846}},
    ]}
    assert resumir_reporte(reporte) == {"n_columnas_drift": 5, "fraccion_drift": 0.3846}


def test_resumir_reporte_sin_metrica_falla():
    with pytest.raises(ValueError, match="DriftedColumnsCount"):
        resumir_reporte({"metrics": []})


@pytest.mark.parametrize("fraccion,alerta", [(0.2, False), (0.3, False), (0.31, True)])
def test_alerta_solo_por_encima_del_umbral(fraccion, alerta):
    assert hay_alerta(fraccion, 0.3) is alerta


def test_solo_el_nivel_06_supera_el_umbral(particion, cfg, cfg_features):
    """Con Evidently: los lotes 0.0 y 0.6 dan reportes distintos y solo el 0.6 genera alerta."""
    entrenamiento, _ = particion
    columnas = seleccionar_variables(entrenamiento, cfg_features, "Exited")
    umbral = cfg["monitoreo"]["umbral_fraccion_drift"]
    fracciones = {}
    for nivel in (0.0, 0.3, 0.6):
        _, resumen = calcular_drift_features(entrenamiento, lote_con_drift(particion, cfg, nivel), columnas)
        fracciones[nivel] = resumen["fraccion_drift"]
        # La lista por columna coincide con el conteo global de Evidently.
        assert len(resumen["columnas_drift"]) == resumen["n_columnas_drift"]
    assert fracciones[0.0] < fracciones[0.3] < fracciones[0.6]
    assert not hay_alerta(fracciones[0.0], umbral)
    assert not hay_alerta(fracciones[0.3], umbral)
    assert hay_alerta(fracciones[0.6], umbral)
