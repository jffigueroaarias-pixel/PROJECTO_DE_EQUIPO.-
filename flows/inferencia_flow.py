"""Pipeline de inferencia con el modelo champion y drift emulado.

1. Ingesta: toma un lote de la parte de prueba del dataset.
2. Drift: aplica el drift emulado con el nivel de config.yaml (drift.nivel).
3. Guarda una copia del lote puntuado, con sus etiquetas reales, en data/lotes/
   para que los pipelines de drift lo lean.
4. Predice con models:/churn-clasificador@champion y guarda las predicciones.

Uso (desde la raíz del repo, con PREFECT_API_URL y MLFLOW_TRACKING_URI definidas):
    poetry run python flows/inferencia_flow.py
"""
import sys
from datetime import datetime
from pathlib import Path

import mlflow
import numpy as np
from mlflow.tracking import MlflowClient
from prefect import flow, get_run_logger, task

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from domain.drift import columnas_afectadas, emular_drift, tomar_lote  # noqa: E402
from domain.modelo import dividir, predecir  # noqa: E402
from scripts.config import cargar_config, ruta  # noqa: E402
from scripts.load_data import cargar_datos  # noqa: E402


@task
def cargar_champion(cfg: dict):
    logger = get_run_logger()
    nombre = cfg["mlflow"]["modelo_registrado"]
    alias = cfg["registro"]["alias"]
    version = MlflowClient().get_model_version_by_alias(nombre, alias)
    modelo = mlflow.sklearn.load_model(f"models:/{nombre}@{alias}")
    logger.info(f"Modelo cargado: {nombre}@{alias} (versión {version.version})")
    return modelo, {"version": version.version, "run_id": version.run_id}


@task
def ingesta_con_drift(cfg: dict):
    """Lote de la parte de prueba con el drift emulado del nivel configurado."""
    logger = get_run_logger()
    cfg_drift = cfg["drift"]
    objetivo = cfg["datos"]["objetivo"]
    rng = np.random.default_rng(cfg_drift["semilla"])

    entrenamiento, prueba = dividir(
        cargar_datos(), objetivo, cfg["division"]["test_size"], cfg["division"]["random_state"]
    )
    lote = tomar_lote(prueba, objetivo, cfg["inferencia"]["tamano_lote"], cfg_drift["nivel"],
                      cfg_drift["subida_tasa_max"], rng)
    lote = emular_drift(lote, entrenamiento, cfg_drift["nivel"], cfg_drift, rng)

    afectadas = columnas_afectadas(cfg_drift["nivel"], cfg_drift["candidatas"])
    logger.info(
        f"Lote de {len(lote)} clientes | nivel de drift {cfg_drift['nivel']} | "
        f"columnas desplazadas: {afectadas or 'ninguna'} | "
        f"tasa de churn: {lote[objetivo].mean():.1%} (entrenamiento: {entrenamiento[objetivo].mean():.1%})"
    )
    return lote


@task
def guardar_lote(lote, cfg: dict, marca: str) -> Path:
    """Copia del lote que se puntúa, con las etiquetas, para los pipelines de drift."""
    carpeta = ruta(cfg["inferencia"]["carpeta_lotes"])
    carpeta.mkdir(parents=True, exist_ok=True)
    archivo = carpeta / f"lote_{marca}_nivel{cfg['drift']['nivel']}.csv"
    lote.to_csv(archivo, index=False)
    get_run_logger().info(f"Lote guardado en {archivo.relative_to(ruta('.'))}")
    return archivo


@task
def puntuar(modelo, lote, umbral: float):
    return predecir(modelo, lote, umbral)


@task
def guardar_predicciones(resultado, cfg: dict, champion: dict, marca: str, archivo_lote: Path) -> Path:
    logger = get_run_logger()
    carpeta = ruta(cfg["inferencia"]["carpeta_salida"])
    carpeta.mkdir(parents=True, exist_ok=True)
    archivo = carpeta / f"prediccion_{marca}.csv"
    # En producción la etiqueta real no se conoce al predecir: no va en el archivo de predicciones.
    resultado.drop(columns=[cfg["datos"]["objetivo"]]).to_csv(archivo, index=False)

    with mlflow.start_run(run_name=f"inferencia-{marca}"):
        mlflow.set_tags({
            "fase": "inferencia",
            "modelo_version": champion["version"],
            "modelo_run_id": champion["run_id"],
            "nivel_drift": cfg["drift"]["nivel"],
            "lote": archivo_lote.name,
        })
        mlflow.log_metrics({
            "filas_predichas": len(resultado),
            "probabilidad_media": resultado["probabilidad_churn"].mean(),
            "tasa_churn_predicha": resultado["prediccion"].mean(),
        })
        mlflow.log_artifact(str(archivo))

    logger.info(
        f"{len(resultado)} predicciones guardadas en {archivo.relative_to(ruta('.'))} | "
        f"churn predicho: {resultado['prediccion'].mean():.1%}"
    )
    return archivo


@flow(name="inferencia-churn", log_prints=True)
def inferencia_flow():
    cfg = cargar_config()
    mlflow.set_experiment(cfg["mlflow"]["experimento"])
    marca = datetime.now().strftime("%Y-%m-%d_%H%M%S")

    modelo, champion = cargar_champion(cfg)
    lote = ingesta_con_drift(cfg)
    archivo_lote = guardar_lote(lote, cfg, marca)
    resultado = puntuar(modelo, lote, cfg["metricas"]["umbral"])
    return str(guardar_predicciones(resultado, cfg, champion, marca, archivo_lote))


if __name__ == "__main__":
    inferencia_flow()
