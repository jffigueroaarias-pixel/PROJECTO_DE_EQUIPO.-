import os
import yaml
import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient
from prefect import flow, task
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

@task(name="Ingesta de Datos Real")
def ingesta_datos():
    print("🤖 Paso 1: Iniciando la ingesta de datos desde el CSV...")
    ruta_csv = "data/clientes.csv"
    if not os.path.exists(ruta_csv):
        raise FileNotFoundError(f"❌ Error crítico: No se encontró el archivo en {ruta_csv}")
    df = pd.read_csv(ruta_csv)
    return df

@task(name="Entrenamiento K-Means e Inferencia de AUC")
def entrenar_y_evaluar_kmeans(df, config):
    print("🏋️‍♂️ Paso 2: Entrenando K-Means y calculando métricas de evaluación...")
    X = df[['edad', 'ingresos_mensuales', 'score_credito']]
    y = (df['score_credito'] > 600).astype(int)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    n_clusters = config.get("kmeans_clusters", 3)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    
    X_augmented = pd.DataFrame(X_scaled, columns=X.columns)
    X_augmented['cluster'] = clusters
    
    clf = LogisticRegression()
    clf.fit(X_augmented, y)
    preds = clf.predict_proba(X_augmented)[:, 1]
    
    auc_score = roc_auc_score(y, preds) if len(set(y)) > 1 else 1.0
    print(f"📊 Entrenamiento finalizado. AUC alcanzado: {auc_score:.4f}")
    return kmeans, auc_score, str(X.shape)

@task(name="Registro en MLflow y Evaluación de Champion")
def registrar_en_mlflow(modelo_kmeans, auc_score, n_features, config):
    print("👑 Paso 3: Conectando con MLflow para registrar el modelo...")
    mlflow.set_experiment("Experimento_KMeans_SantoTo")
    
    with mlflow.start_run() as run:
        mlflow.set_tag("fase", "entrenamiento")
        mlflow.log_param("features", "NUMERIC")
        mlflow.log_param("n_features", n_features)
        mlflow.log_metric("auc", auc_score)
        
        # Registrar el modelo de forma persistente en MLflow
        model_name = config.get("model_name", "modelo_clientes_kmeans")
        
        # Corregido: Registrar el modelo directamente al guardarlo
        mlflow.sklearn.log_model(
            sk_model=modelo_kmeans, 
            artifact_path="model",
            registered_model_name=model_name
        )
        
        client = MlflowClient()
        auc_minimo = config.get("auc_minimo", 0.70)
        
        try:
            # Intentar ver si hay un champion anterior
            version_champion = client.get_model_version_by_alias(model_name, "champion")
            run_champion = client.get_run(version_champion.run_id)
            auc_champion = float(run_champion.data.metrics["auc"])
            
            if auc_score >= auc_champion:
                client.set_registered_model_alias(model_name, "champion", "1")
                print("🏆 ¡Modelo coronado como Champion por superar al anterior!")
            else:
                print("🛡️ El Champion anterior sigue siendo mejor.")
        except Exception:
            # Coronar la versión 1 si es la primera vez que corre en limpio
            client.set_registered_model_alias(model_name, "champion", "1")
            print("🥇 Modelo coronado con éxito como el primer Champion oficial del sistema.")

@flow(name="Pipeline de Entrenamiento Grupo")
def mi_primer_pipeline():
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    datos = ingesta_datos()
    modelo, auc, num_features = entrenar_y_evaluar_kmeans(datos, config)
    registrar_en_mlflow(modelo, auc, num_features, config)

if __name__ == "__main__":
    mi_primer_pipeline()
