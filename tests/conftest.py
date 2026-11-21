"""Datos y configuración compartidos por los tests. Ningún test necesita MLflow ni Prefect."""
import numpy as np
import pytest

from domain.modelo import dividir
from scripts.config import cargar_config
from scripts.load_data import cargar_datos


@pytest.fixture(scope="session")
def cfg():
    return cargar_config()


@pytest.fixture(scope="session")
def cfg_features():
    return cargar_config("features.yaml")


@pytest.fixture(scope="session")
def df():
    return cargar_datos()


@pytest.fixture(scope="session")
def particion(df, cfg):
    return dividir(df, cfg["datos"]["objetivo"], cfg["division"]["test_size"], cfg["division"]["random_state"])


@pytest.fixture
def rng():
    return np.random.default_rng(0)


class ModeloFalso:
    """Sustituye al champion: probabilidad fija, sin MLflow."""

    def __init__(self, probabilidad=0.7):
        self.probabilidad = probabilidad

    def predict_proba(self, X):
        return np.column_stack([np.full(len(X), 1 - self.probabilidad), np.full(len(X), self.probabilidad)])


@pytest.fixture
def modelo_falso():
    return ModeloFalso()
