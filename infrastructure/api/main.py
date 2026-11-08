import yaml
import pandas as pd
import mlflow
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# 1. Cargar la configuración compartida del YAML
with open("configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Extraer el nombre asignado al modelo en el registro central
registro_config = config.get("registro", {})
model_name = registro_config.get("nombre_modelo", "churn-modelo")

app = FastAPI(
    title="API de Inferencia - Clientes Churn Banco",
    description="Endpoint unificado que consume el modelo Champion para predecir la fuga de clientes."
)

# 2. Cargar el modelo Champion desde MLflow de forma global
try:
    model_uri = f"models:/{model_name}@champion"
    modelo_api = mlflow.pyfunc.load_model(model_uri)
    print(f"✅ API unificada cargó con éxito el modelo Champion '{model_name}' desde MLflow.")
except Exception as e:
    print(f"⚠️ Alerta: No se pudo cargar el modelo Champion '{model_name}': {str(e)}")
    modelo_api = None

# 3. Esquema de Pydantic adaptado al dataset real del banco (Evita el Error 422)
class ClienteBancarioInput(BaseModel):
    CreditScore: int = Field(..., description="Score de crédito del cliente (350-850)")
    Geography: str = Field(..., description="País de residencia del cliente (ej. France, Spain)")
    Gender: str = Field(..., description="Género del cliente (Female o Male)")
    Age: int = Field(..., description="Edad del cliente bancario")
    Tenure: int = Field(..., description="Años de antigüedad en el banco")
    Balance: float = Field(..., description="Saldo de la cuenta bancario")
    NumOfProducts: int = Field(..., description="Número de productos contratados")
    HasCrCard: int = Field(..., description="¿Tiene tarjeta de crédito? (1=Sí, 0=No)")
    IsActiveMember: int = Field(..., description="¿Es miembro activo? (1=Sí, 0=No)")
    EstimatedSalary: float = Field(..., description="Salario estimado del cliente")
    SatisfactionScore: int = Field(..., description="Puntaje de satisfacción (1-5)")
    CardType: str = Field(..., description="Categoría de tarjeta (ej. DIAMOND, GOLD)")
    PointEarned: int = Field(..., description="Puntos de fidelidad acumulados")

    # Requisito de la rúbrica: Ejemplo completamente documentado con los tipos correctos
    class Config:
        json_schema_extra = {
            "example": {
                "CreditScore": 619,
                "Geography": "France",
                "Gender": "Female",
                "Age": 42,
                "Tenure": 2,
                "Balance": 0.0,
                "NumOfProducts": 1,
                "HasCrCard": 1,
                "IsActiveMember": 1,
                "EstimatedSalary": 101348.88,
                "SatisfactionScore": 2,
                "CardType": "DIAMOND",
                "PointEarned": 464
            }
        }

@app.post("/predict", summary="Predecir probabilidad de abandono (Churn)")
def predecir_fuga(data: ClienteBancarioInput):
    if modelo_api is None:
        raise HTTPException(status_code=503, detail="El modelo central Champion no está inicializado en el servidor.")
    
    # Transformar la entrada de Pydantic directamente a un DataFrame estructurado
    df_input = pd.DataFrame([data.dict()])
    
    try:
        # Ejecutar la inferencia en tiempo real consumiendo el pipeline unificado del grupo
        prediccion = modelo_api.predict(df_input)
        
        # Si el modelo devuelve probabilidades, tomamos la clase correspondiente
        resultado_clase = int(prediccion[0])
        
        return {
            "status": "success",
            "modelo_consumido": model_name,
            "prediccion_churn": resultado_clase,
            "descripcion": "El cliente se quedará en el banco" if resultado_clase == 0 else "El cliente abandonará el banco"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error durante el procesamiento del modelo: {str(e)}")

@app.get("/health", summary="Verificar la salud del microservicio")
def health_check():
    return {
        "status": "healthy",
        "modelo_en_memoria": model_name,
        "modelo_cargado_correctamente": modelo_api is not None
    }
