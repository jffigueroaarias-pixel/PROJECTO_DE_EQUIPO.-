"""Pipeline de entrenamiento: ingesta -> entrenamiento -> evaluación -> registro.

Entrena cada candidato de config.yaml con las variables de features.yaml,
registra un run por candidato en MLflow (tag fase=entrenamiento) y promueve
al mejor a champion solo si supera el AUC mínimo y al champion vigente.

Uso (desde la raíz del repo, con PREFECT_API_URL y MLFLOW_TRACKING_URI definidas):
    poetry run python flows/entrenamiento_flow.py
"""
import sys
import warnings
from pathlib import Path

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient
from prefect import flow, get_run_logger, task
from prefect.artifacts import create_markdown_artifact

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from domain.modelo import calcular_metricas, construir_pipeline, dividir, seleccionar_variables  # noqa: E402
from scripts.config import cargar_config  # noqa: E402
from scripts.load_data import cargar_datos  # noqa: E402

# Aviso de MLflow sobre columnas enteras sin nulos: no aplica, el dataset no tiene nulos.
warnings.filterwarnings("ignore", message="Hint: Inferred schema contains integer")


@task
def ingesta(cfg: dict, cfg_features: dict):
    logger = get_run_logger()
    df = cargar_datos()
    objetivo = cfg["datos"]["objetivo"]
    columnas = seleccionar_variables(df, cfg_features, objetivo)

    # Misma división que usan la inferencia y el monitoreo de drift.
    entrenamiento_df, prueba_df = dividir(df, objetivo, cfg["division"]["test_size"], cfg["division"]["random_state"])
    X_train, y_train = entrenamiento_df[columnas], entrenamiento_df[objetivo]
    X_test, y_test = prueba_df[columnas], prueba_df[objetivo]
    logger.info(
        f"Dataset: {len(df)} filas | features={cfg_features['seleccion']} "
        f"({len(columnas)} variables) | train={len(X_train)} test={len(X_test)}"
    )
    return X_train, X_test, y_train, y_test, columnas


@task
def entrenamiento(candidato: dict, X_train, y_train, columnas: list[str]):
    logger = get_run_logger()
    modelo = construir_pipeline(X_train, columnas, candidato["tipo"], candidato["parametros"])
    modelo.fit(X_train, y_train)
    logger.info(f"Entrenado {candidato['nombre_run']} ({candidato['tipo']})")
    return modelo


@task
def evaluacion(candidato: dict, modelo, X_train, X_test, y_test, cfg: dict, cfg_features: dict, columnas: list[str]) -> dict:
    """Calcula las métricas en test y deja el run completo en MLflow."""
    logger = get_run_logger()
    probabilidades = modelo.predict_proba(X_test)[:, 1]
    metricas = calcular_metricas(y_test, probabilidades, cfg["metricas"]["umbral"])

    with mlflow.start_run(run_name=candidato["nombre_run"]) as run:
        mlflow.set_tags({"fase": "entrenamiento", "tipo_modelo": candidato["tipo"]})
        mlflow.log_params(candidato["parametros"])
        mlflow.log_params({
            "features": cfg_features["seleccion"],
            "n_features": len(columnas),
            "columnas": ",".join(columnas),
            "umbral": cfg["metricas"]["umbral"],
            "test_size": cfg["division"]["test_size"],
        })
        mlflow.log_metrics(metricas)
        # MLflow guarda el modelo con skops, que por seguridad rechaza los árboles
        # (random forest, gradient boosting) salvo que se marquen como confiables.
        # Lo son: los entrenamos nosotros en este mismo run.
        info_modelo = mlflow.sklearn.log_model(
            modelo,
            name="modelo",
            input_example=X_train.head(3),
            skops_trusted_types=["sklearn.tree._tree.Tree"],
        )

    logger.info(f"{candidato['nombre_run']}: AUC={metricas['auc']:.4f} recall={metricas['recall']:.4f}")
    return {
        "nombre_run": candidato["nombre_run"],
        "run_id": run.info.run_id,
        "model_uri": info_modelo.model_uri,
        **metricas,
    }


def auc_champion_actual(client: MlflowClient, nombre_modelo: str, alias: str) -> float | None:
    """AUC del champion vigente, o None si todavía no hay champion."""
    try:
        version = client.get_model_version_by_alias(nombre_modelo, alias)
    except MlflowException:
        return None
    return client.get_run(version.run_id).data.metrics.get("auc")


@task
def registro(resultados: list[dict], cfg: dict) -> str:
    """Promueve al mejor candidato a champion solo si mejora lo que ya hay."""
    logger = get_run_logger()
    reg = cfg["registro"]
    nombre_modelo = cfg["mlflow"]["modelo_registrado"]
    client = MlflowClient()

    mejor = max(resultados, key=lambda r: r["auc"])
    auc_champion = auc_champion_actual(client, nombre_modelo, reg["alias"])
    logger.info(
        f"Mejor candidato: {mejor['nombre_run']} (AUC {mejor['auc']:.4f}) | "
        f"champion vigente: {'ninguno' if auc_champion is None else f'AUC {auc_champion:.4f}'}"
    )

    if mejor["auc"] < reg["auc_minimo"]:
        decision = f"No promovido: AUC {mejor['auc']:.4f} < mínimo {reg['auc_minimo']}"
    elif auc_champion is not None and mejor["auc"] <= auc_champion + reg["mejora_minima"]:
        decision = (
            f"No promovido: AUC {mejor['auc']:.4f} no supera al champion "
            f"({auc_champion:.4f} + mejora mínima {reg['mejora_minima']})"
        )
    else:
        version = mlflow.register_model(mejor["model_uri"], nombre_modelo)
        client.set_registered_model_alias(nombre_modelo, reg["alias"], version.version)
        client.set_model_version_tag(nombre_modelo, version.version, "estado", "aprobado")
        decision = f"Versión {version.version} ({mejor['nombre_run']}) es el nuevo {reg['alias']} con AUC {mejor['auc']:.4f}"

    with mlflow.start_run(run_id=mejor["run_id"]):
        mlflow.set_tag("decision_registro", decision)
    logger.info(decision)
    return decision


@flow(name="entrenamiento-churn", log_prints=True)
def entrenamiento_flow():
    cfg = cargar_config()
    cfg_features = cargar_config("features.yaml")
    # MLflow toma el servidor de la variable de entorno MLFLOW_TRACKING_URI.
    mlflow.set_experiment(cfg["mlflow"]["experimento"])

    X_train, X_test, y_train, y_test, columnas = ingesta(cfg, cfg_features)

    resultados = []
    for candidato in cfg["candidatos"]:
        modelo = entrenamiento(candidato, X_train, y_train, columnas)
        resultados.append(evaluacion(candidato, modelo, X_train, X_test, y_test, cfg, cfg_features, columnas))

    decision = registro(resultados, cfg)

    tabla = "\n".join(
        f"| {r['nombre_run']} | {r['auc']:.4f} | {r['recall']:.4f} | {r['f1']:.4f} |" for r in resultados
    )
    create_markdown_artifact(
        key="entrenamiento-churn",
        markdown=(
            f"## Entrenamiento churn\n\n"
            f"**Features:** {cfg_features['seleccion']} ({len(columnas)} variables)\n\n"
            f"| Run | AUC | Recall | F1 |\n|---|---|---|---|\n{tabla}\n\n"
            f"**Decisión:** {decision}\n"
        ),
    )
    return decision


if __name__ == "__main__":
    entrenamiento_flow()
