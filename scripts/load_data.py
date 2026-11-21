"""Carga del dataset de churn con validación de columnas.

Origen: Customer-Churn-Records.csv, del repositorio del curso
(actividades/data/). Es pequeño (~800 KB), así que se versiona en data/.

Uso: poetry run python scripts/load_data.py
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.config import cargar_config, ruta  # noqa: E402

COLUMNAS_ESPERADAS = [
    "RowNumber", "CustomerId", "Surname", "CreditScore", "Geography", "Gender",
    "Age", "Tenure", "Balance", "NumOfProducts", "HasCrCard", "IsActiveMember",
    "EstimatedSalary", "Exited", "Complain", "Satisfaction Score", "Card Type",
    "Point Earned",
]


def cargar_datos(ruta_csv: str | Path | None = None) -> pd.DataFrame:
    """Lee el CSV y comprueba que tenga las columnas esperadas y sin nulos en el objetivo."""
    if ruta_csv is None:
        ruta_csv = ruta(cargar_config()["datos"]["ruta"])
    ruta_csv = Path(ruta_csv)
    if not ruta_csv.exists():
        raise FileNotFoundError(f"No se encontró el dataset en {ruta_csv}")

    df = pd.read_csv(ruta_csv)

    faltantes = set(COLUMNAS_ESPERADAS) - set(df.columns)
    if faltantes:
        raise ValueError(f"Faltan columnas en el dataset: {sorted(faltantes)}")
    if df["Exited"].isna().any():
        raise ValueError("La columna objetivo 'Exited' tiene valores nulos")

    return df


if __name__ == "__main__":
    df = cargar_datos()
    print(f"Dataset cargado: {df.shape[0]} filas x {df.shape[1]} columnas")
    print(f"Tasa de churn: {df['Exited'].mean():.2%}")
