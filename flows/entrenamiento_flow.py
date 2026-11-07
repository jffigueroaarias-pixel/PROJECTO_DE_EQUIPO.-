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

@task(name="Ingesta de Datos Real Banco")
def ingesta_datos():
    print("🤖 Paso 1: Iniciando la ingesta del dataset oficial del banco...")
    ruta_csv = "data/clientes.csv"
    if not os.path.exists(ruta_csv):
        raise FileNotFoundError(f"❌ Error crítico: No se encontró el archivo en {ruta_csv}")
    
    df = pd.read_csv(ruta_csv)
    if 'target' not in df.columns and 'Exited' in df.columns:
        df = df.rename(columns={'Exited': 'target'})
        
    print(f"✅ Dataset cargado correctamente. Registros bancarios detectados: {df.shape}")
    return df

@task(name="Entrenamiento K-Means con Clientes del Banco")
def entrenar_y_evaluar_kmeans(df, config):
    print("🏋️‍♂️ Paso 2: Procesando variables numéricas bancarias y entrenando K-Means...")
    columnas_numericas = ['CreditScore', 'Age', 'Balance', 'Tenure', 'EstimatedSalary']
    X = df[columnas_numericas]
    y = df['target']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    n_clusters = config.get("kmeans_clusters", 4)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    
    X_augmented = pd.DataFrame(X_scaled, columns=X.columns)
    X_augmented['cluster'] = clusters
    
    clf = LogisticRegression()
    clf.fit(X_augmented, y)
    preds = clf.predict_proba(X_augmented)[:, 1]
    
    auc_score = roc_auc_score(y, preds)
    print(f"📊 K-Means finalizado con {n_clusters} grupos. AUC del clasificador: {auc_score:.4f}")
    return kmeans, auc_score, str(X.shape)

@task(name="Registro en Model Registry de MLflow")
def registrar_en_mlflow(modelo_kmeans, auc_score, n_features, config):
    print("👑 Paso 3: Sincronizando con el servidor de MLflow...")
    mlflow.set_experiment("Experimento_KMeans_SantoTo")
    
    with mlflow.start_run() as run:
        mlflow.set_tag("fase", "entrenamiento")
        mlflow.log_param("features", "NUMERIC")
        mlflow.log_param("n_features", n_features)
        mlflow.log_metric("auc", auc_score)
        
        model_name = config.get("model_name", "modelo_clientes_kmeans")
        mlflow.sklearn.log_model(
            sk_model=modelo_kmeans, 
            artifact_path="model",
            registered_model_name=model_name
        )
        
        client = MlflowClient()
        try:
            client.set_registered_model_alias(model_name, "champion", "1")
            print("🥇 Modelo guardado con éxito como el Champion oficial del banco.")
        except Exception:
            print("⚠️ El modelo se registró pero hubo un problema asignando el alias.")

@flow(name="Pipeline de Entrenamiento Grupo")
def mi_primer_pipeline():
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    datos = ingesta_datos()
    modelo, auc, num_features = entrenar_y_evaluar_kmeans(datos, config)
    registrar_en_mlflow(modelo, auc, num_features, config)

if __name__ == "__main__":
    mi_primer_pipeline()
