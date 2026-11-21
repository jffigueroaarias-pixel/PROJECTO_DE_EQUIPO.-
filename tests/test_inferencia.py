"""Tests de la inferencia: predicción con umbral y lectura de lotes guardados."""
import pandas as pd
import pytest

from domain.modelo import predecir
from scripts.lotes import resolver_lote
from tests.conftest import ModeloFalso


def test_predecir_anade_probabilidad_y_prediccion(df, modelo_falso):
    resultado = predecir(modelo_falso, df.head(10), umbral=0.5)
    assert {"probabilidad_churn", "prediccion"} <= set(resultado.columns)
    assert (resultado["probabilidad_churn"] == 0.7).all()
    assert (resultado["prediccion"] == 1).all()


def test_umbral_decide_la_prediccion(df):
    resultado = predecir(ModeloFalso(0.4), df.head(5), umbral=0.5)
    assert (resultado["prediccion"] == 0).all()
    resultado = predecir(ModeloFalso(0.4), df.head(5), umbral=0.3)
    assert (resultado["prediccion"] == 1).all()


def test_predecir_no_modifica_el_lote(df, modelo_falso):
    lote = df.head(5)
    predecir(modelo_falso, lote, 0.5)
    assert "probabilidad_churn" not in lote.columns


def test_resolver_lote_toma_el_mas_reciente(tmp_path, monkeypatch, cfg):
    monkeypatch.setattr("scripts.lotes.ruta", lambda relativa: tmp_path / relativa)
    carpeta = tmp_path / cfg["inferencia"]["carpeta_lotes"]
    carpeta.mkdir(parents=True)
    for nombre in ["lote_2026-11-20_080000_nivel0.0.csv", "lote_2026-11-21_090000_nivel0.6.csv"]:
        pd.DataFrame({"a": [1]}).to_csv(carpeta / nombre, index=False)
    assert resolver_lote(cfg).name == "lote_2026-11-21_090000_nivel0.6.csv"


def test_resolver_lote_sin_lotes_falla(tmp_path, monkeypatch, cfg):
    monkeypatch.setattr("scripts.lotes.ruta", lambda relativa: tmp_path / relativa)
    with pytest.raises(FileNotFoundError, match="inferencia_flow"):
        resolver_lote(cfg)
