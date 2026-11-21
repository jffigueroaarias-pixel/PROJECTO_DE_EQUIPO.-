"""Emulación y detección de drift. Sin MLflow ni Prefect.

Emulación (algoritmo del grupo)
-------------------------------
Con un nivel entre 0.0 y 1.0:

1. Target: el lote se muestrea de la parte de prueba de forma estratificada,
   de modo que la tasa de churn sea tasa_base + nivel * subida_tasa_max.
   Las etiquetas siguen siendo reales (no se invierten), así que el lote
   conserva la relación entre variables y objetivo.
2. Variables: se desplazan las primeras round(nivel * len(candidatas)) columnas
   de la lista `candidatas` del YAML, en ese orden (no al azar):
   - numéricas: se suma nivel * desplazamiento_max_std desviaciones estándar
     (más nivel => más columnas afectadas y desplazamientos mayores);
   - categóricas: una fracción `nivel` de las filas pasa a la categoría
     configurada (por ejemplo, Geography -> Germany).

Detección
---------
- Features: Evidently (DataDriftPreset) compara cada variable del lote con
  los datos de entrenamiento y devuelve la fracción de columnas con drift.
- Evento: prueba chi-cuadrado de la tasa de la variable objetivo del lote
  frente a la de entrenamiento.
"""
import numpy as np
import pandas as pd
from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset
from scipy.stats import chi2_contingency


def validar_nivel(nivel: float) -> float:
    if not 0.0 <= nivel <= 1.0:
        raise ValueError(f"El nivel de drift debe estar entre 0.0 y 1.0 (recibido: {nivel})")
    return float(nivel)


def tomar_lote(prueba: pd.DataFrame, objetivo: str, tamano: int, nivel: float, subida_tasa_max: float,
               rng: np.random.Generator) -> pd.DataFrame:
    """Muestra de la parte de prueba con la tasa de churn desplazada según el nivel."""
    nivel = validar_nivel(nivel)
    positivos = prueba[prueba[objetivo] == 1]
    negativos = prueba[prueba[objetivo] == 0]
    tasa = min(prueba[objetivo].mean() + nivel * subida_tasa_max, 1.0)
    n_pos = min(round(tasa * tamano), len(positivos))
    n_neg = min(tamano - n_pos, len(negativos))

    lote = pd.concat([
        positivos.sample(n=n_pos, random_state=rng.integers(1_000_000)),
        negativos.sample(n=n_neg, random_state=rng.integers(1_000_000)),
    ])
    # Se mezcla para que los positivos no queden todos al principio.
    return lote.sample(frac=1, random_state=rng.integers(1_000_000)).reset_index(drop=True)


def columnas_afectadas(nivel: float, candidatas: list[str]) -> list[str]:
    return candidatas[: round(validar_nivel(nivel) * len(candidatas))]


def emular_drift(lote: pd.DataFrame, referencia: pd.DataFrame, nivel: float, cfg_drift: dict,
                 rng: np.random.Generator) -> pd.DataFrame:
    """Desplaza la distribución de las columnas afectadas por el nivel. No modifica `lote`."""
    salida = lote.copy()
    categorias = cfg_drift.get("categoria_destino", {})
    for columna in columnas_afectadas(nivel, cfg_drift["candidatas"]):
        if columna in categorias:
            filas = rng.random(len(salida)) < nivel
            salida.loc[filas, columna] = categorias[columna]
        else:
            desplazamiento = nivel * cfg_drift["desplazamiento_max_std"] * referencia[columna].std()
            nuevo = salida[columna] + desplazamiento
            # Las columnas enteras (edad, antigüedad...) siguen siendo enteras.
            if pd.api.types.is_integer_dtype(lote[columna]):
                nuevo = nuevo.round().astype(lote[columna].dtype)
            salida[columna] = nuevo
    return salida


def validar_etiquetas(lote: pd.DataFrame, objetivo: str) -> None:
    """El drift de evento necesita las etiquetas reales del lote."""
    if objetivo not in lote.columns:
        raise ValueError(
            f"El lote no trae la columna '{objetivo}' con las etiquetas reales: "
            "no se puede calcular el drift de evento. Espera a que lleguen las etiquetas."
        )
    faltantes = int(lote[objetivo].isna().sum())
    if faltantes:
        raise ValueError(
            f"El lote tiene {faltantes} filas sin etiqueta en '{objetivo}': "
            "no se puede calcular el drift de evento hasta tener todas las etiquetas."
        )


def drift_evento(y_referencia: pd.Series, y_lote: pd.Series, alfa: float) -> dict:
    """Chi-cuadrado sobre la tabla 2x2 (positivos/negativos en referencia y lote)."""
    tabla = [
        [int((y_referencia == 1).sum()), int((y_referencia == 0).sum())],
        [int((y_lote == 1).sum()), int((y_lote == 0).sum())],
    ]
    _, p_valor, _, _ = chi2_contingency(tabla)
    return {
        "tasa_referencia": float(y_referencia.mean()),
        "tasa_lote": float(y_lote.mean()),
        "diferencia": float(y_lote.mean() - y_referencia.mean()),
        "p_valor": float(p_valor),
        "drift": bool(p_valor < alfa),
    }


def calcular_drift_features(referencia: pd.DataFrame, lote: pd.DataFrame, columnas: list[str]):
    """Ejecuta Evidently y devuelve (reporte, resumen). El reporte se puede guardar en HTML."""
    numericas = [c for c in columnas if pd.api.types.is_numeric_dtype(referencia[c])]
    categoricas = [c for c in columnas if c not in numericas]
    definicion = DataDefinition(numerical_columns=numericas, categorical_columns=categoricas)
    reporte = Report([DataDriftPreset()]).run(
        reference_data=Dataset.from_pandas(referencia[columnas], data_definition=definicion),
        current_data=Dataset.from_pandas(lote[columnas], data_definition=definicion),
    )
    como_dict = reporte.dict()
    return reporte, {**resumir_reporte(como_dict), "columnas_drift": columnas_con_drift(como_dict)}


def resumir_reporte(reporte: dict) -> dict:
    """Extrae del diccionario de Evidently el número y la fracción de columnas con drift."""
    for metrica in reporte["metrics"]:
        if metrica["metric_name"].startswith("DriftedColumnsCount"):
            return {
                "n_columnas_drift": int(metrica["value"]["count"]),
                "fraccion_drift": float(metrica["value"]["share"]),
            }
    raise ValueError("El reporte de Evidently no contiene la métrica DriftedColumnsCount")


def columnas_con_drift(reporte: dict) -> list[str]:
    """Columnas que Evidently marcó con drift (métricas ValueDrift).

    Con métodos de distancia (Wasserstein, Jensen-Shannon) hay drift si el valor
    alcanza el umbral; con pruebas estadísticas (p-valor), si queda por debajo.
    """
    columnas = []
    for metrica in reporte["metrics"]:
        config = metrica.get("config", {})
        if not config.get("type", "").endswith("ValueDrift"):
            continue
        es_distancia = "distance" in config.get("method", "").lower()
        valor, umbral = metrica["value"], config["threshold"]
        if (valor >= umbral) if es_distancia else (valor < umbral):
            columnas.append(config["column"])
    return columnas


def hay_alerta(fraccion: float, umbral: float) -> bool:
    return fraccion > umbral
