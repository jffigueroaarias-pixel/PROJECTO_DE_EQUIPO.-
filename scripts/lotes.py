"""Acceso a los lotes guardados por la inferencia y a los datos de referencia."""
from pathlib import Path

import pandas as pd

from domain.modelo import dividir
from scripts.config import ruta
from scripts.load_data import cargar_datos


def resolver_lote(cfg: dict, ruta_lote: str | None = None) -> Path:
    """Ruta del lote indicado o, si no se indica, del más reciente de data/lotes/."""
    if ruta_lote:
        archivo = Path(ruta_lote)
        archivo = archivo if archivo.is_absolute() else ruta(ruta_lote)
        if not archivo.exists():
            raise FileNotFoundError(f"No existe el lote {archivo}")
        return archivo

    lotes = sorted(ruta(cfg["inferencia"]["carpeta_lotes"]).glob("lote_*.csv"))
    if not lotes:
        raise FileNotFoundError("No hay lotes guardados: ejecuta primero flows/inferencia_flow.py")
    # El nombre empieza por la fecha y hora, así que el orden alfabético es el cronológico.
    return lotes[-1]


def datos_referencia(cfg: dict) -> pd.DataFrame:
    """Parte de entrenamiento del dataset: la distribución contra la que se compara cada lote."""
    entrenamiento, _ = dividir(
        cargar_datos(), cfg["datos"]["objetivo"], cfg["division"]["test_size"], cfg["division"]["random_state"]
    )
    return entrenamiento
