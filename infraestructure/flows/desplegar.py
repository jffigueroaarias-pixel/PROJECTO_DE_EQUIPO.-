import sys
from pathlib import Path

# Asegurar ruta raíz
ROOT_DIR = Path("/workspaces/PROJECTO_DE_EQUIPO.-")
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from infraestructure.flows.entrenamiento_flow import entrenamiento_flow
from infraestructure.flows.inferencia_flow import inferencia_flow

def desplegar():
    # Crear despliegue de entrenamiento con la sintaxis moderna local
    entrenamiento_deployment = entrenamiento_flow.to_deployment(
        name="entrenamiento-churn",
        work_pool_name="default-agent-pool"
    )
    
    # Crear despliegue de inferencia programado (diario a las 06:00 UTC)
    inferencia_deployment = inferencia_flow.to_deployment(
        name="inferencia-diaria",
        schedule={"cron": "0 6 * * *"},
        work_pool_name="default-agent-pool"
    )

    # Registrar los despliegues en el servidor local de Prefect
    entrenamiento_deployment.apply()
    inferencia_deployment.apply()

    print("¡Despliegues de Prefect registrados exitosamente!")

if __name__ == "__main__":
    desplegar()