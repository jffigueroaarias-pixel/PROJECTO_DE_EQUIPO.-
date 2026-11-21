"""API de inferencia que sirve el modelo champion.

Carga el champion al arrancar. Si el entrenamiento promueve un champion
nuevo, hay que reiniciar la API para que lo use.

Uso (desde la raíz del repo, con MLFLOW_TRACKING_URI definida):
    poetry run uvicorn api.main:app --host 0.0.0.0 --port 8000
Documentación interactiva: http://localhost:8000/docs
"""
from contextlib import asynccontextmanager

import mlflow
import pandas as pd
from fastapi import FastAPI
from mlflow.tracking import MlflowClient
from pydantic import BaseModel, ConfigDict, Field

from domain.modelo import predecir
from scripts.config import cargar_config

cfg = cargar_config()
estado: dict = {}


def cargar_champion():
    """Devuelve (modelo, versión) del champion. Los tests la sustituyen por un modelo falso."""
    nombre = cfg["mlflow"]["modelo_registrado"]
    alias = cfg["registro"]["alias"]
    version = MlflowClient().get_model_version_by_alias(nombre, alias)
    return mlflow.sklearn.load_model(f"models:/{nombre}@{alias}"), version.version


@asynccontextmanager
async def lifespan(app: FastAPI):
    estado["modelo"], estado["version"] = cargar_champion()
    yield
    estado.clear()


app = FastAPI(
    title="API de churn",
    description="Predice la probabilidad de que un cliente abandone el banco, con el modelo champion de MLflow.",
    lifespan=lifespan,
)


class Cliente(BaseModel):
    # Algunas columnas del CSV tienen espacios; el JSON usa esos mismos nombres.
    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "example": {
                "CreditScore": 619, "Geography": "France", "Gender": "Female", "Age": 42,
                "Tenure": 2, "Balance": 0.0, "NumOfProducts": 1, "HasCrCard": 1,
                "IsActiveMember": 1, "EstimatedSalary": 101348.88,
                "Satisfaction Score": 2, "Card Type": "DIAMOND", "Point Earned": 464,
            }
        },
    )

    CreditScore: int
    Geography: str
    Gender: str
    Age: int
    Tenure: int
    Balance: float
    NumOfProducts: int
    HasCrCard: int
    IsActiveMember: int
    EstimatedSalary: float
    SatisfactionScore: int = Field(alias="Satisfaction Score")
    CardType: str = Field(alias="Card Type")
    PointEarned: int = Field(alias="Point Earned")


class Prediccion(BaseModel):
    probabilidad_churn: float
    prediccion: int
    version_modelo: str


@app.get("/health")
def health():
    return {"estado": "ok", "modelo": cfg["mlflow"]["modelo_registrado"], "version_modelo": estado.get("version")}


@app.post("/predict", response_model=Prediccion)
def predict(cliente: Cliente):
    fila = predecir(estado["modelo"], pd.DataFrame([cliente.model_dump(by_alias=True)]), cfg["metricas"]["umbral"])
    return Prediccion(
        probabilidad_churn=round(float(fila["probabilidad_churn"].iloc[0]), 4),
        prediccion=int(fila["prediccion"].iloc[0]),
        version_modelo=str(estado["version"]),
    )
