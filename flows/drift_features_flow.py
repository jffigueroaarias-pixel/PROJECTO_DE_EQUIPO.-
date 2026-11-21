"""Pipeline de drift de features con Evidently.

Compara la distribución de cada variable del modelo en un lote con la de los
datos de entrenamiento, guarda el reporte HTML del lote y emite una alerta si
la fracción de variables con drift supera monitoreo.umbral_fraccion_drift.

Uso (desde la raíz del repo, con PREFECT_API_URL y MLFLOW_TRACKING_URI definidas):
    poetry run python flows/drift_features_flow.py                 # último lote
    poetry run python flows/drift_features_flow.py data/lotes/X.csv # un lote concreto
"""
import json
import sys
from pathlib import Path

import mlflow
import pandas as pd
from prefect import flow, get_run_logger, task
from prefect.artifacts import create_markdown_artifact

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from domain.drift import calcular_drift_features, hay_alerta  # noqa: E402
from domain.modelo import seleccionar_variables  # noqa: E402
from scripts.config import cargar_config, ruta  # noqa: E402
from scripts.lotes import datos_referencia, resolver_lote  # noqa: E402


@task
def cargar(cfg: dict, ruta_lote: str | None):
    archivo = resolver_lote(cfg, ruta_lote)
    referencia = datos_referencia(cfg)
    columnas = seleccionar_variables(referencia, cargar_config("features.yaml"), cfg["datos"]["objetivo"])
    get_run_logger().info(f"Lote: {archivo.name} | referencia: {len(referencia)} filas | {len(columnas)} variables")
    return archivo, pd.read_csv(archivo), referencia, columnas


@task
def calcular_y_guardar(archivo: Path, lote, referencia, columnas: list[str], cfg: dict) -> dict:
    reporte, resumen = calcular_drift_features(referencia, lote, columnas)
    carpeta = ruta(cfg["monitoreo"]["carpeta_reportes"])
    carpeta.mkdir(parents=True, exist_ok=True)
    html = carpeta / f"drift_features_{archivo.stem}.html"
    reporte.save_html(str(html))
    resumen_json = carpeta / f"drift_features_{archivo.stem}.json"
    resumen_json.write_text(json.dumps({"lote": archivo.name, **resumen}, indent=2), encoding="utf-8")
    get_run_logger().info(f"Reporte guardado en {html.relative_to(ruta('.'))}")
    return {**resumen, "html": html, "json": resumen_json}


@task
def alertar(resumen: dict, umbral: float, archivo: Path) -> bool:
    logger = get_run_logger()
    alerta = hay_alerta(resumen["fraccion_drift"], umbral)
    mensaje = (
        f"{resumen['n_columnas_drift']} variables con drift {resumen['columnas_drift']} "
        f"(fracción {resumen['fraccion_drift']:.2f}, umbral {umbral}) en {archivo.name}"
    )
    if alerta:
        logger.warning(f"ALERTA DE DRIFT DE FEATURES: {mensaje}")
    else:
        logger.info(f"Sin alerta: {mensaje}")
    return alerta


@flow(name="drift-features-churn", log_prints=True)
def drift_features_flow(ruta_lote: str | None = None) -> bool:
    cfg = cargar_config()
    umbral = cfg["monitoreo"]["umbral_fraccion_drift"]
    mlflow.set_experiment(cfg["mlflow"]["experimento"])

    archivo, lote, referencia, columnas = cargar(cfg, ruta_lote)
    resumen = calcular_y_guardar(archivo, lote, referencia, columnas, cfg)
    alerta = alertar(resumen, umbral, archivo)

    with mlflow.start_run(run_name=f"drift-features-{archivo.stem}"):
        mlflow.set_tags({"fase": "drift_features", "lote": archivo.name, "alerta": alerta})
        mlflow.log_params({"umbral_fraccion_drift": umbral, "n_variables": len(columnas)})
        mlflow.log_metrics({
            "fraccion_drift": resumen["fraccion_drift"],
            "n_columnas_drift": resumen["n_columnas_drift"],
        })
        mlflow.log_artifact(str(resumen["html"]))
        mlflow.log_artifact(str(resumen["json"]))

    create_markdown_artifact(
        key="drift-features",
        markdown=(
            f"## Drift de features\n\n**Lote:** {archivo.name}\n\n"
            f"| Variables con drift | Fracción | Umbral | Alerta |\n|---|---|---|---|\n"
            f"| {resumen['n_columnas_drift']} de {len(columnas)} | {resumen['fraccion_drift']:.2f} | "
            f"{umbral} | {'SÍ' if alerta else 'no'} |\n\n"
            f"Reporte: `{resumen['html'].relative_to(ruta('.'))}`\n"
        ),
    )
    return alerta


if __name__ == "__main__":
    drift_features_flow(sys.argv[1] if len(sys.argv) > 1 else None)
