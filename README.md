# Proyecto MLOps 

Proyecto del curso de MLOps.  

**Integrantes:** [Nombre 1], [Nombre 2], [Nombre 3]

## Requisitos
- Python 3.12 a 3.14
- Poetry 2.x
- Git
- Docker (para MLflow y Prefect)

## Estructura
```
configs/     configuración del proyecto
flows/       flows de Prefect
scripts/     scripts de datos (carga / generación)
tests/       pruebas con pytest
notebooks/   exploración
docs/        documentación
```

## Instalación del entorno
```bash
git clone <URL_DEL_REPOSITORIO>.git
cd <NOMBRE_DEL_REPO>
poetry install
poetry run python -c "import prefect, mlflow, sklearn; print('ok')"
```

## Levantar servicios
```bash
cd services
docker compose up -d
```
- MLflow: http://localhost:5000
- Prefect: http://localhost:4200

## Ejecutar el flow de prueba
```bash
export PREFECT_API_URL="http://localhost:4200/api"
export MLFLOW_TRACKING_URI="http://localhost:5000"
poetry run python flows/prueba_flow.py
```

## Pruebas
```bash
poetry run pytest -q
```
## Levantar servicios

**En un computador local:**
```bash
cd services
docker compose up -d
```

**En GitHub Codespaces** (usa el archivo de ajustes para que las UIs funcionen por el puerto reenviado):
```bash
cd services
docker compose -f docker-compose.yml -f docker-compose.codespaces.yml up -d
```

- MLflow: puerto 5000 (en Codespaces, ábrelo desde la pestaña **Ports**)
- Prefect: puerto 4200