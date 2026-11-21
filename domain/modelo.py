"""Lógica del modelo de churn: selección de variables, pipeline y métricas.

No depende de Prefect ni de MLflow, así que se puede probar y reutilizar
por separado (flows, API, tests).
"""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

MODELOS = {
    "logistic_regression": LogisticRegression,
    "random_forest": RandomForestClassifier,
    "gradient_boosting": GradientBoostingClassifier,
}


def dividir(df: pd.DataFrame, objetivo: str, test_size: float, random_state: int):
    """División estratificada y reproducible en entrenamiento y prueba.

    La comparten el entrenamiento, la inferencia (que puntúa lotes de la parte
    de prueba) y el monitoreo (que usa la de entrenamiento como referencia).
    """
    return train_test_split(df, test_size=test_size, random_state=random_state, stratify=df[objetivo])


def predecir(modelo, lote: pd.DataFrame, umbral: float) -> pd.DataFrame:
    """Añade al lote la probabilidad de churn y la predicción 0/1."""
    resultado = lote.copy()
    resultado["probabilidad_churn"] = modelo.predict_proba(lote)[:, 1]
    resultado["prediccion"] = (resultado["probabilidad_churn"] >= umbral).astype(int)
    return resultado


def seleccionar_variables(df: pd.DataFrame, cfg_features: dict, objetivo: str) -> list[str]:
    """Devuelve las columnas que usará el modelo según el alias de `features.yaml`."""
    excluir = set(cfg_features.get("excluir", [])) | {objetivo}
    disponibles = [c for c in df.columns if c not in excluir]
    seleccion = cfg_features["seleccion"].upper()

    if seleccion == "ALL":
        return disponibles
    if seleccion == "NUMERIC":
        return [c for c in disponibles if pd.api.types.is_numeric_dtype(df[c])]
    if seleccion == "CUSTOM":
        columnas = cfg_features.get("custom") or []
        desconocidas = [c for c in columnas if c not in df.columns]
        if desconocidas:
            raise ValueError(f"Columnas de 'custom' que no existen en el CSV: {desconocidas}")
        prohibidas = [c for c in columnas if c in excluir]
        if prohibidas:
            raise ValueError(f"Columnas de 'custom' que están en 'excluir': {prohibidas}")
        return list(columnas)
    raise ValueError(f"Selección desconocida: {seleccion!r}. Usa ALL, NUMERIC o CUSTOM")


def construir_pipeline(df: pd.DataFrame, columnas: list[str], tipo: str, parametros: dict) -> Pipeline:
    """Preprocesado (escalado + one-hot) y clasificador en un solo objeto.

    El ColumnTransformer descarta cualquier columna que no esté en `columnas`,
    así que el modelo entrenado acepta la fila completa de un cliente.
    """
    if tipo not in MODELOS:
        raise ValueError(f"Tipo de modelo desconocido: {tipo!r}. Opciones: {list(MODELOS)}")

    numericas = [c for c in columnas if pd.api.types.is_numeric_dtype(df[c])]
    categoricas = [c for c in columnas if c not in numericas]

    preprocesado = ColumnTransformer(
        [
            ("num", StandardScaler(), numericas),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categoricas),
        ],
        remainder="drop",
    )
    return Pipeline([("preprocesado", preprocesado), ("clasificador", MODELOS[tipo](**parametros))])


def calcular_metricas(y_real, probabilidades, umbral: float) -> dict[str, float]:
    predicciones = (probabilidades >= umbral).astype(int)
    return {
        "auc": roc_auc_score(y_real, probabilidades),
        "accuracy": accuracy_score(y_real, predicciones),
        "precision": precision_score(y_real, predicciones, zero_division=0),
        "recall": recall_score(y_real, predicciones, zero_division=0),
        "f1": f1_score(y_real, predicciones, zero_division=0),
    }
