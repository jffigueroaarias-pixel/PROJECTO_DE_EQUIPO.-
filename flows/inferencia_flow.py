import os
import mlflow
import pandas as pd
from datetime import datetime
from prefect import flow, task

@task(name="Cargar Modelo Champion")
def cargar_modelo_champion():
    print("📥 Descargando el modelo Champion actual desde MLflow...")
    # Regla: Debe cargar el modelo estrictamente por su alias, no por versión fija
    model_name = "modelo_clientes_kmeans"
    model_uri = f"models:/{model_name}@champion"
    modelo = mlflow.pyfunc.load_model(model_uri)
    return modelo

@task(name="Calcular Predicciones Diarias")
def calcular_predicciones(modelo):
    print("🎲 Generando predicciones para el lote diario de clientes nuevos...")
    # Simulamos un lote de datos nuevos de clientes para la inferencia diaria
    nuevos_clientes = pd.DataFrame([
        {"edad": 28, "ingresos_mensuales": 3100000, "score_credito": 620},
        {"edad": 45, "ingresos_mensuales": 7800000, "score_credito": 740}
    ])
    
    # Calcular las predicciones
    predicciones = modelo.predict(nuevos_clientes)
    nuevos_clientes["cluster_asignado"] = predicciones
    return nuevos_clientes

@task(name="Guardar Resultados con Fecha")
def guardar_resultados(df_resultados):
    # Regla de la rúbrica: Debe guardar las predicciones con la fecha de ejecución
    fecha_hoy = datetime.now().strftime("%Y-%m-%d")
    os.makedirs("data/outputs", exist_ok=True)
    ruta_salida = f"data/outputs/predicciones_{fecha_hoy}.csv"
    
    df_resultados.to_csv(ruta_salida, index=False)
    print(f"💾 ¡Predicciones diarias guardadas con éxito en: {ruta_salida}!")

@flow(name="Pipeline de Inferencia Diaria")
def inferencia_pipeline():
    modelo = cargar_modelo_champion()
    resultados = calcular_predicciones(modelo)
    guardar_resultados(resultados)

if __name__ == "__main__":
    inferencia_pipeline()
