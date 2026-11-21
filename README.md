# Proyecto MLOps 

Proyecto del curso de MLOps.  

**Integrantes:** [Juan Jose Rodriguez], [Jorge Emmanuel Marin Linares], [Jose Fernando Figueroa]

### Actividad 1 – Fundamentos, versionamiento y seguimiento

| Entregable | Dónde |
|---|---|
| Entorno reproducible con Poetry | [`pyproject.toml`](pyproject.toml), [`poetry.lock`](poetry.lock) |
| Servicios MLflow y Prefect (Docker) | [`services/`](services/) (incluye el ajuste para Codespaces) |
| Flow de validación | [`flows/prueba_flow.py`](flows/prueba_flow.py) |

### Actividad 2 – Pipelines de entrenamiento e inferencia, despliegue y API

| Entregable | Dónde |
|---|---|
| Pipeline de entrenamiento (ingesta → entrenamiento → evaluación → registro) | [`flows/entrenamiento_flow.py`](flows/entrenamiento_flow.py) |
| Regla del champion (AUC mínimo y mejora sobre el vigente) | `registro()` en [`flows/entrenamiento_flow.py`](flows/entrenamiento_flow.py) |
| Pipeline de inferencia con el modelo `champion` | [`flows/inferencia_flow.py`](flows/inferencia_flow.py) |
| Despliegues en Prefect (inferencia diaria) | [`flows/desplegar.py`](flows/desplegar.py) |
| API FastAPI con predicción de ejemplo | [`api/main.py`](api/main.py), [`docs/actividad-2.md`](docs/actividad-2.md) |
| Configuración separada del código | [`configs/config.yaml`](configs/config.yaml), [`configs/features.yaml`](configs/features.yaml) |
| Lógica del modelo, sin Prefect ni MLflow | [`domain/modelo.py`](domain/modelo.py) |

### Actividad 3 – Monitoreo de drift, pruebas e integración continua

| Entregable | Dónde |
|---|---|
| Drift emulado configurable (0.0, 0.3, 0.6) y su documentación | sección `drift` de [`configs/config.yaml`](configs/config.yaml), [`domain/drift.py`](domain/drift.py), [`docs/actividad-3.md`](docs/actividad-3.md) |
| Ingesta de lotes de la parte de prueba con drift | [`flows/inferencia_flow.py`](flows/inferencia_flow.py) |
| Drift de features con Evidently (reporte por lote) | [`flows/drift_features_flow.py`](flows/drift_features_flow.py) |
| Drift de evento (target) y rechazo de lotes sin etiquetas | [`flows/drift_evento_flow.py`](flows/drift_evento_flow.py) |
| Umbral configurable y alerta | sección `monitoreo` de [`configs/config.yaml`](configs/config.yaml) |
| Tests de datos, modelo, inferencia, API y drift (64) | [`tests/`](tests/) |
| Integración continua (tests en cada Pull Request) | [`.github/workflows/tests.yml`](.github/workflows/tests.yml), pestaña [Actions](https://github.com/jffigueroaarias-pixel/PROJECTO_DE_EQUIPO.-/actions) |
| Reportes de drift de ejemplo y Memoria 3 | [`docs/evidencias/`](docs/evidencias/) |



## Cómo ejecutarlo

```bash
# 1. Entorno
poetry install

# 2. Servicios (en Codespaces, con el archivo de ajustes)
cd services
docker compose -f docker-compose.yml -f docker-compose.codespaces.yml up -d
cd ..

# 3. Variables de entorno (en cada terminal)
export PREFECT_API_URL="http://localhost:4200/api"
export MLFLOW_TRACKING_URI="http://localhost:5000"

# 4. Pipelines
poetry run python flows/entrenamiento_flow.py      # entrena y promueve el champion
poetry run python flows/inferencia_flow.py         # lote con el drift de drift.nivel
poetry run python flows/drift_features_flow.py     
poetry run python flows/drift_evento_flow.py       

# 5. API (documentación en /docs)
poetry run uvicorn api.main:app --host 0.0.0.0 --port 8000

# 6. Tests
poetry run pytest
```

Interfaces: MLflow en el puerto 5000 y Prefect en el 4200. Más detalle en
[`docs/actividad-2.md`](docs/actividad-2.md) y [`docs/actividad-3.md`](docs/actividad-3.md).
