import yaml
import pandas as pd
import mlflow
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# 1. Cargar la configuración del YAML
with open("configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)

model_name = config.get("model_name", "modelo_clientes_kmeans")

app = FastAPI(
    title="API de Inferencia Grupo - KMeans SantoTomás",
    description="Endpoint para predecir el clúster de un cliente en tiempo real usando el modelo Champion."
)

# 2. Intentar cargar el modelo Champion globalmente al encender la API
try:
    model_uri = f"models:/{model_name}@champion"
    modelo_api = mlflow.pyfunc.load_model(model_uri)
    print(f"✅ API cargó con éxito el modelo Champion desde MLflow.")
except Exception as e:
    print(f"❌ Error al cargar el modelo Champion: {str(e)}")
    modelo_api = None

# 3. Estructura de entrada requerida para el cliente
class ClienteInput(BaseModel):
    edad: float
    ingresos_mensuales: float
    score_credito: float

    # Requisito de la rúbrica: Incluir una predicción de ejemplo en la documentación
    class Config:
        json_schema_extra = {
            "example": {
                "edad": 35.0,
                "ingresos_mensuales": 4200000.0,
                "score_credito": 680.0
            }
        }

@app.post("/predict", summary="Predecir el grupo del cliente")
def predecir_cliente(data: ClienteInput):
    if modelo_api is None:
        raise HTTPException(status_code=503, detail="El modelo Champion no está disponible en el servidor.")
    
    # Convertir la entrada en un DataFrame de Pandas para el modelo
    df_input = pd.DataFrame([data.dict()])
    
    # Calcular la predicción del clúster
    prediccion = modelo_api.predict(df_input)
    
    return {
        "status": "success",
        "modelo_utilizado": model_name,
        "cluster_asignado": int(prediccion[0])
    }

@app.get("/health", summary="Verificar estado de la API")
def health_check():
    return {"status": "healthy", "modelo_cargado": modelo_api is not None}
