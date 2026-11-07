import os
import sys
from pathlib import Path

# Forzar la ruta absoluta de la raíz del proyecto para evitar errores de niveles
ROOT_DIR = Path("/workspaces/PROJECTO_DE_EQUIPO.-")
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from datetime import datetime
import yaml
import mlflow
import mlflow.sklearn
from prefect import flow, task
from scripts.load_data import cargar_datos_churn
from domain.modelo import seleccionar_features, construir_pipeline, evaluar_modelo

# Configurar URIs
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


@task(name="cargar_champion_y_datos")
def tarea_cargar_champion_y_datos():
    """Carga el modelo champion de MLflow y el dataset completo."""
    client = mlflow.tracking.MlflowClient()
    
    # Cargar configuración para obtener el nombre del modelo
    config_path = Path("configs/config.yaml")
    if not config_path.exists():
        config_path = Path("../configs/config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    nombre_modelo = config["registro"]["nombre_modelo"]
    tamano_lote = config["inferencia"]["tamano_lote"]
    
    try:
        # Cargar modelo por su alias champion
        model_uri = f"models:/{nombre_modelo}@champion"
        model = mlflow.sklearn.load_model(model_uri)
        
        # Obtener información de la versión champion para trazabilidad
        champion_version = client.get_model_version_by_alias(nombre_modelo, "champion")
        run_id = champion_version.run_id
    except Exception as e:
        raise RuntimeError(
            f"No se pudo cargar el modelo champion '{nombre_modelo}@champion'. "
            f"Asegúrate de haber ejecutado el entrenamiento antes. Detalle: {e}"
        )
        
    df = cargar_datos_churn()
    
    # Tomar un lote al azar simulando clientes nuevos
    df_lote = df.sample(n=min(tamano_lote, len(df)), random_state=None)
    
    return model, df_lote, run_id, champion_version.version


@task(name="ejecutar_inferencia_y_guardar")
def tarea_ejecutar_inferencia_y_guardar(model, df_lote, run_id, version_num):
    """Calcula las predicciones con el modelo champion y guarda el archivo CSV."""
    mlflow.set_experiment("churn-training-mlops")
    
    # Predecir probabilidades y clases
    # El modelo es un pipeline que ya incluye preprocesamiento, por lo que acepta el DataFrame completo
    X_pred = df_lote.drop(columns=["Exited"], errors="ignore")
    
    y_prob = model.predict_proba(X_pred)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    
    # Construir DataFrame de resultados
    df_resultados = df_lote.copy()
    df_resultados["probabilidad_churn"] = y_prob
    df_resultados["prediccion"] = y_pred
    
    # Guardar en la carpeta data con la fecha actual
    fecha_str = datetime.now().strftime("%Y-%m-%d")
    
    # Asegurar que la carpeta data exista
    data_dir = Path("data")
    if not data_dir.exists():
        data_dir = Path("../data")
        if not data_dir.exists():
            data_dir = Path(".")
            
    archivo_salida = data_dir / f"prediccion_{fecha_str}.csv"
    df_resultados.to_csv(archivo_salida, index=False)
    print(f"Predicciones guardadas en: {archivo_salida}")
    
    # Registrar run en MLflow para la inferencia
    with mlflow.start_run(run_name=f"inferencia-{fecha_str}") as run:
        mlflow.set_tag("fase", "inferencia")
        mlflow.set_tag("modelo_version", str(version_num))
        mlflow.set_tag("modelo_run_id", str(run_id))
        
        mlflow.log_metric("filas_predichas", len(df_lote))
        mlflow.log_metric("probabilidad_media", float(y_prob.mean()))
        
        # Guardar el CSV como artefacto
        mlflow.log_artifact(str(archivo_salida))
        
    return str(archivo_salida)


@flow(name="inferencia-churn")
def inferencia_flow():
    """Flujo principal de inferencia diaria en Prefect."""
    model, df_lote, run_id, version_num = tarea_cargar_champion_y_datos()
    tarea_ejecutar_inferencia_y_guardar(model, df_lote, run_id, version_num)


if __name__ == "__main__":
    inferencia_flow()
