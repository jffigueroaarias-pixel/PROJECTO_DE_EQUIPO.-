"""Registra los despliegues en Prefect y los deja atendiendo ejecuciones.

- entrenamiento-churn/entrenamiento-manual: sin horario, se lanza a mano.
- inferencia-churn/inferencia-diaria: todos los días a las 06:00 (hora de Bogotá).
- drift-features-churn/drift-features-diario: todos los días a las 06:30, sobre el lote de la inferencia.
- drift-evento-churn/drift-evento-manual: sin horario, porque necesita las etiquetas
  reales del lote, que en producción llegan con retraso.

Uso (desde la raíz del repo, con PREFECT_API_URL y MLFLOW_TRACKING_URI definidas):
    poetry run python flows/desplegar.py
La terminal debe quedarse abierta: si se cierra, Prefect no puede lanzar las ejecuciones.
"""
import sys
from pathlib import Path

from prefect import serve
from prefect.schedules import Cron

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flows.drift_evento_flow import drift_evento_flow  # noqa: E402
from flows.drift_features_flow import drift_features_flow  # noqa: E402
from flows.entrenamiento_flow import entrenamiento_flow  # noqa: E402
from flows.inferencia_flow import inferencia_flow  # noqa: E402

ZONA = "America/Bogota"

if __name__ == "__main__":
    serve(
        entrenamiento_flow.to_deployment(name="entrenamiento-manual", tags=["churn", "entrenamiento"]),
        inferencia_flow.to_deployment(
            name="inferencia-diaria", schedule=Cron("0 6 * * *", timezone=ZONA), tags=["churn", "inferencia"]
        ),
        drift_features_flow.to_deployment(
            name="drift-features-diario", schedule=Cron("30 6 * * *", timezone=ZONA), tags=["churn", "drift"]
        ),
        drift_evento_flow.to_deployment(name="drift-evento-manual", tags=["churn", "drift"]),
    )
