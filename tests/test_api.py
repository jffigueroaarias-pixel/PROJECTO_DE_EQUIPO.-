"""Tests de la API con un modelo falso: no necesitan MLflow."""
import pytest
from fastapi.testclient import TestClient

import api.main
from tests.conftest import ModeloFalso

CLIENTE = {
    "CreditScore": 619, "Geography": "France", "Gender": "Female", "Age": 42, "Tenure": 2,
    "Balance": 0.0, "NumOfProducts": 1, "HasCrCard": 1, "IsActiveMember": 1,
    "EstimatedSalary": 101348.88, "Satisfaction Score": 2, "Card Type": "DIAMOND", "Point Earned": 464,
}


@pytest.fixture
def cliente_http(monkeypatch):
    # Sustituye la carga del champion desde MLflow por un modelo falso, versión 7.
    monkeypatch.setattr(api.main, "cargar_champion", lambda: (ModeloFalso(0.7), "7"))
    with TestClient(api.main.app) as cliente:  # el `with` ejecuta el arranque (lifespan)
        yield cliente


def test_health(cliente_http):
    respuesta = cliente_http.get("/health")
    assert respuesta.status_code == 200
    assert respuesta.json()["version_modelo"] == "7"


def test_predict(cliente_http):
    respuesta = cliente_http.post("/predict", json=CLIENTE)
    assert respuesta.status_code == 200
    assert respuesta.json() == {"probabilidad_churn": 0.7, "prediccion": 1, "version_modelo": "7"}


def test_predict_campo_faltante_devuelve_422(cliente_http):
    incompleto = {k: v for k, v in CLIENTE.items() if k != "Age"}
    respuesta = cliente_http.post("/predict", json=incompleto)
    assert respuesta.status_code == 422
    assert "Age" in respuesta.text


def test_predict_tipo_invalido_devuelve_422(cliente_http):
    respuesta = cliente_http.post("/predict", json={**CLIENTE, "Age": "cuarenta"})
    assert respuesta.status_code == 422
