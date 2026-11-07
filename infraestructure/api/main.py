import os
from pathlib import Path
import yaml
import mlflow
import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Configurar URIs de MLflow
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

app = FastAPI(
    title="API de Predicción de Churn",
    description="Sirve el modelo champion registrado en MLflow con alias 'champion'.",
    version="1.0.0",
)

# Variable global para el modelo champion y su versión
model_champion = None
champion_version_num = "Desconocida"


def cargar_champion():
    global model_champion, champion_version_num
    client = mlflow.tracking.MlflowClient()
    
    config_path = Path("configs/config.yaml")
    if not config_path.exists():
        config_path = Path("../configs/config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    nombre_modelo = config["registro"]["nombre_modelo"]
    
    try:
        model_uri = f"models:/{nombre_modelo}@champion"
        model_champion = mlflow.sklearn.load_model(model_uri)
        
        ver = client.get_model_version_by_alias(nombre_modelo, "champion")
        champion_version_num = str(ver.version)
        print(f"-> Modelo champion versión {champion_version_num} cargado exitosamente en la API.")
    except Exception as e:
        print(f"Advertencia: No se pudo cargar el champion al arrancar la API: {e}")


@app.on_event("startup")
def startup_event():
    cargar_champion()


@app.get("/health")
def health_check():
    """Estado y versión del modelo."""
    return {
        "status": "activo",
        "modelo_champion_cargado": model_champion is not None,
        "version_modelo": champion_version_num,
    }


# Esquema de datos de ejemplo para la documentación de FastAPI
class ClienteInput(BaseModel):
    CreditScore: int = Field(..., example=619)
    Geography: str = Field(..., example="France")
    Gender: str = Field(..., example="Female")
    Age: int = Field(..., example=42)
    Tenure: int = Field(..., example=2)
    Balance: float = Field(..., example=0.0)
    NumOfProducts: int = Field(..., example=1)
    HasCrCard: int = Field(..., example=1)
    IsActiveMember: int = Field(..., example=1)
    EstimatedSalary: float = Field(..., example=101348.88)
    Complain: int = Field(..., example=1)
    Satisfaction_Score: int = Field(2, alias="Satisfaction Score", example=2)
    Card_Type: str = Field("DIAMOND", alias="Card Type", example="DIAMOND")
    Point_Earned: int = Field(464, alias="Point Earned", example=464)

    class Config:
        populate_by_name = True


@app.post("/predict")
def predict(cliente: ClienteInput):
    """Endpoint para predecir si un cliente abandonará (churn)."""
    if model_champion is None:
        raise HTTPException(
            status_code=500,
            detail="El modelo champion no está cargado. Asegúrate de entrenar y registrar un modelo primero."
        )
        
    # Convertir datos de entrada a DataFrame con los nombres originales de columnas
    data_dict = cliente.dict(by_alias=True)
    df_input = pd.DataFrame([data_dict])
    
    try:
        y_prob = float(model_champion.predict_proba(df_input)[:, 1][0])
        y_pred = int(y_prob >= 0.5)
        
        return {
            "probabilidad_churn": y_prob,
            "prediccion": y_pred,
            "version_modelo": champion_version_num,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error al procesar la predicción: {e}")