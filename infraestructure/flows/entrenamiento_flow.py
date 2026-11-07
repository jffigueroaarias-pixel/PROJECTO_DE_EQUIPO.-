import os
import sys
from pathlib import Path

# Forzar la ruta absoluta de la raíz del proyecto para evitar errores de niveles
ROOT_DIR = Path("/workspaces/PROJECTO_DE_EQUIPO.-")
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

import yaml
import mlflow
import mlflow.sklearn
from prefect import flow, task
from scripts.load_data import cargar_datos_churn
from domain.modelo import seleccionar_features, construir_pipeline, evaluar_modelo

# Configurar URIs de MLflow y Prefect desde el entorno o valores por defecto
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


@task(name="cargar_datos")
def tarea_cargar_datos():
    """Carga los datos usando el script dedicado."""
    df = cargar_datos_churn()
    return df


@task(name="entrenar_y_registrar")
def tarea_entrenar_y_registrar(df, candidato, feature_config, config):
    """Entrena un modelo para un conjunto de features y registra el run en MLflow."""
    mlflow.set_experiment("churn-training-mlops")
    
    nombre_candidato = candidato["nombre_run"]
    tipo_modelo = candidato["tipo"]
    parametros = candidato.get("parametros", {})
    
    nombre_features = feature_config["nombre"]
    columnas_features = feature_config["columnas"]
    
    # Seleccionar features
    if columnas_features == "ALL":
        X = seleccionar_features(df, tipo_features="ALL")
    elif columnas_features == "NUMERIC":
        X = seleccionar_features(df, tipo_features="NUMERIC")
    else:
        X = seleccionar_features(df, tipo_features="CUSTOM", custom_cols=columnas_features)
        
    y = df["Exited"]
    
    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=config["modelo"]["semilla"]
    )
    
    run_name = f"{nombre_candidato}-{nombre_features.lower()}"
    
    with mlflow.start_run(run_name=run_name) as run:
        # Registrar tag de fase
        mlflow.set_tag("fase", "entrenamiento")
        
        # Registrar parámetros
        mlflow.log_param("modelo", tipo_modelo)
        mlflow.log_param("features", nombre_features)
        mlflow.log_param("n_features", X.shape[1])
        mlflow.log_param("semilla", config["modelo"]["semilla"])
        for p, val in parametros.items():
            mlflow.log_param(p, val)
            
        # Construir y entrenar pipeline
        pipeline = construir_pipeline(tipo_modelo, parametros, X_train)
        pipeline.fit(X_train, y_train)
        
        # Evaluar
        umbral_metrica = config["metricas"]["umbral"]
        metrics = evaluar_modelo(pipeline, X_test, y_test, umbral=umbral_metrica)
        
        for m_name, m_val in metrics.items():
            mlflow.log_metric(m_name, m_val)
            
        # Guardar modelo como artefacto / registro  
        mlflow.sklearn.log_model(
            pipeline, 
            name="modelo",
            serialization_format="cloudpickle"
        )
        
        run_id = run.info.run_id
        auc = metrics["auc"]
        
    return run_id, auc, tipo_modelo, nombre_features


@task(name="promover_si_mejora")
def tarea_promover_si_mejora(resultados_entrenamiento, config):
    """Compara los candidatos con el champion actual y promueve si supera los umbrales."""
    client = mlflow.tracking.MlflowClient()
    nombre_modelo = config["registro"]["nombre_modelo"]
    umbral_min_auc = config["registro"]["umbral_auc"]
    
    # Encontrar el mejor run de esta ejecución
    mejor_run = max(resultados_entrenamiento, key=lambda x: x[1])
    run_id, auc, tipo_modelo, features_name = mejor_run
    
    print(f"\nMejor modelo de la tanda: {tipo_modelo} ({features_name}) con AUC: {auc:.4f}")
    
    if auc < umbral_min_auc:
        print(f"-> El AUC ({auc:.4f}) no supera el umbral mínimo requerido ({umbral_min_auc}). No se registra como champion.")
        return
        
    # Obtener el champion actual si existe
    champion_auc = 0.0
    try:
        champion_version = client.get_model_version_by_alias(nombre_modelo, "champion")
        # Obtener run_id del champion anterior para ver su AUC
        champ_run = client.get_run(champion_version.run_id)
        champion_auc = float(champ_run.data.metrics.get("auc", 0.0))
        print(f"-> Champion actual tiene AUC: {champion_auc:.4f}")
    except Exception:
        print("-> No hay ningún champion registrado actualmente. Este será el primero.")
        champion_auc = 0.0
        
    if auc > champion_auc:
        print(f"-> ¡Nuevo récord! Supera al champion anterior ({auc:.4f} > {champion_auc:.4f}). Registrando nueva versión...")
        
        # Registrar el modelo del run como nueva versión del modelo registrado
        artifact_uri = f"runs:/{run_id}/modelo"
        mv = mlflow.register_model(artifact_uri, nombre_modelo)
        
        # Asignar alias champion
        client.set_registered_model_alias(nombre_modelo, "champion", mv.version)
        
        # Asignar etiqueta de aprobado
        client.set_model_version_tag(nombre_modelo, mv.version, "estado", "aprobado")
        print(f"-> Versión {mv.version} registrada y promovida exitosamente como CHAMPION.")
    else:
        print(f"-> El AUC ({auc:.4f}) no supera al champion vigente ({champion_auc:.4f}). El alias no cambia.")


@flow(name="entrenamiento-churn")
def entrenamiento_flow():
    """Flujo principal de entrenamiento en Prefect."""
    # Cargar configuración desde YAML
    config_path = Path("configs/config.yaml")
    if not config_path.exists():
        config_path = Path("../configs/config.yaml")
        
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    df = tarea_cargar_datos()
    
    candidatos = config["modelo"]["candidatos"]
    features_list = config["features"]
    
    resultados = []
    for candidato in candidatos:
        for feat_conf in features_list:
            res = tarea_entrenar_y_registrar(df, candidato, feat_conf, config)
            resultados.append(res)
            
    tarea_promover_si_mejora(resultados, config)


if __name__ == "__main__":
    from sklearn.model_selection import train_test_split
    entrenamiento_flow()