"""Pipeline de drift de evento (target).

Compara la tasa de churn (Exited) de un lote con la de los datos de
entrenamiento mediante una prueba chi-cuadrado. Necesita las etiquetas reales
del lote, que en producción llegan con retraso: si faltan, el flow falla con
un mensaje claro en lugar de calcular nada.

Uso (desde la raíz del repo, con PREFECT_API_URL y MLFLOW_TRACKING_URI definidas):
    poetry run python flows/drift_evento_flow.py                 # último lote
    poetry run python flows/drift_evento_flow.py data/lotes/X.csv # un lote concreto
"""
import json
import sys
from pathlib import Path

import mlflow
import pandas as pd
from prefect import flow, get_run_logger, task
from prefect.artifacts import create_markdown_artifact

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from domain.drift import drift_evento, validar_etiquetas  # noqa: E402
from scripts.config import cargar_config, ruta  # noqa: E402
from scripts.lotes import datos_referencia, resolver_lote  # noqa: E402


@task
def cargar_lote_etiquetado(cfg: dict, ruta_lote: str | None):
    archivo = resolver_lote(cfg, ruta_lote)
    lote = pd.read_csv(archivo)
    validar_etiquetas(lote, cfg["datos"]["objetivo"])
    return archivo, lote


@task
def comparar_target(lote, cfg: dict, archivo: Path) -> dict:
    objetivo = cfg["datos"]["objetivo"]
    resultado = drift_evento(datos_referencia(cfg)[objetivo], lote[objetivo], cfg["monitoreo"]["alfa_evento"])

    carpeta = ruta(cfg["monitoreo"]["carpeta_reportes"])
    carpeta.mkdir(parents=True, exist_ok=True)
    salida = carpeta / f"drift_evento_{archivo.stem}.json"
    salida.write_text(json.dumps({"lote": archivo.name, **resultado}, indent=2), encoding="utf-8")

    logger = get_run_logger()
    mensaje = (
        f"tasa de churn {resultado['tasa_lote']:.1%} en el lote vs {resultado['tasa_referencia']:.1%} "
        f"en entrenamiento (p-valor {resultado['p_valor']:.4f}, alfa {cfg['monitoreo']['alfa_evento']})"
    )
    if resultado["drift"]:
        logger.warning(f"ALERTA DE DRIFT DE EVENTO: {mensaje}")
    else:
        logger.info(f"Sin drift de evento: {mensaje}")
    return {**resultado, "json": salida}


@flow(name="drift-evento-churn", log_prints=True)
def drift_evento_flow(ruta_lote: str | None = None) -> bool:
    cfg = cargar_config()
    mlflow.set_experiment(cfg["mlflow"]["experimento"])

    archivo, lote = cargar_lote_etiquetado(cfg, ruta_lote)
    resultado = comparar_target(lote, cfg, archivo)

    with mlflow.start_run(run_name=f"drift-evento-{archivo.stem}"):
        mlflow.set_tags({"fase": "drift_evento", "lote": archivo.name, "alerta": resultado["drift"]})
        mlflow.log_param("alfa_evento", cfg["monitoreo"]["alfa_evento"])
        mlflow.log_metrics({
            "tasa_referencia": resultado["tasa_referencia"],
            "tasa_lote": resultado["tasa_lote"],
            "p_valor": resultado["p_valor"],
        })
        mlflow.log_artifact(str(resultado["json"]))

    create_markdown_artifact(
        key="drift-evento",
        markdown=(
            f"## Drift de evento\n\n**Lote:** {archivo.name}\n\n"
            f"| Tasa entrenamiento | Tasa lote | p-valor | Drift |\n|---|---|---|---|\n"
            f"| {resultado['tasa_referencia']:.1%} | {resultado['tasa_lote']:.1%} | "
            f"{resultado['p_valor']:.4f} | {'SÍ' if resultado['drift'] else 'no'} |\n"
        ),
    )
    return resultado["drift"]


if __name__ == "__main__":
    drift_evento_flow(sys.argv[1] if len(sys.argv) > 1 else None)
