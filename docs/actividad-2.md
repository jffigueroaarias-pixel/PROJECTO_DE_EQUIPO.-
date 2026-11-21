# Actividad 2: pipelines de entrenamiento e inferencia, y API

Caso de uso: **churn bancario** (`data/Customer-Churn-Records.csv`, 10.000 clientes, objetivo `Exited`).

## Estructura

```
configs/config.yaml          datos, candidatos, umbrales, MLflow, inferencia
configs/features.yaml        selección de variables: ALL | NUMERIC | CUSTOM
data/                        dataset (predicciones/ no se versiona)
domain/modelo.py             variables, pipeline y métricas (sin Prefect ni MLflow)
scripts/config.py            lectura de los YAML
scripts/load_data.py         carga del CSV con validación de columnas
flows/entrenamiento_flow.py  ingesta -> entrenamiento -> evaluación -> registro
flows/inferencia_flow.py     predicciones diarias con el champion
flows/desplegar.py           despliegues en Prefect
api/main.py                  API FastAPI que sirve el champion
tests/test_modelo.py         pruebas de datos y modelo
```

## Variables de entorno

En **cada** terminal donde se ejecute un flow, el despliegue o la API:

```bash
export PREFECT_API_URL="http://localhost:4200/api"
export MLFLOW_TRACKING_URI="http://localhost:5000"
```

## 1. Entrenamiento

```bash
poetry run python flows/entrenamiento_flow.py
```

Entrena los candidatos de `config.yaml` con las variables de `features.yaml`. Cada candidato deja un run en el
experimento `churn-mlops-grupo` con el tag `fase = entrenamiento`, los parámetros `features` y `n_features`, y las
métricas AUC, accuracy, precision, recall y F1.

El mejor candidato pasa a ser `champion` del modelo `churn-clasificador` **solo si**:

1. su AUC es mayor o igual que `registro.auc_minimo`, y
2. su AUC supera al del champion vigente más `registro.mejora_minima`.

Si no cumple, el run queda registrado (con el tag `decision_registro` explicando por qué) y el alias no cambia.

**Decisión de diseño: `Complain` se excluye.** Coincide con `Exited` en 9.966 de 10.000 filas. Con ella el AUC sube a
~0.999, pero es una fuga de datos: de un cliente nuevo aún no sabemos si se quejó. Se excluye en `features.yaml`.

## 2. Inferencia

```bash
poetry run python flows/inferencia_flow.py
```

Carga `models:/churn-clasificador@champion`, toma `inferencia.tamano_lote` clientes al azar y guarda
`data/predicciones/prediccion_<fecha>.csv` con `probabilidad_churn` y `prediccion`. Registra un run
`inferencia-<fecha>` con el tag `fase = inferencia`, la versión del modelo y el CSV como artefacto.

## 3. Despliegues

```bash
poetry run python flows/desplegar.py
```

Deja la terminal abierta. En Prefect, **Deployments** muestra:

- `entrenamiento-churn/entrenamiento-manual`: sin horario, se lanza con **Run**.
- `inferencia-churn/inferencia-diaria`: todos los días a las 06:00 (America/Bogota).

Lanzar uno a mano desde otra terminal:

```bash
poetry run prefect deployment run "inferencia-churn/inferencia-diaria"
```

> En Codespaces, el entorno se detiene tras un tiempo sin uso. El horario queda registrado en Prefect, pero las
> ejecuciones programadas solo se lanzan mientras el Codespace y `desplegar.py` estén activos.

## 4. API

```bash
poetry run uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Documentación interactiva en `/docs` (en Codespaces, puerto 8000 desde la pestaña **Ports**).
Si el entrenamiento promueve un champion nuevo, reinicia la API.

**Predicción de ejemplo:**

```bash
curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"CreditScore": 619, "Geography": "France", "Gender": "Female", "Age": 42, "Tenure": 2,
       "Balance": 0.0, "NumOfProducts": 1, "HasCrCard": 1, "IsActiveMember": 1,
       "EstimatedSalary": 101348.88, "Satisfaction Score": 2, "Card Type": "DIAMOND",
       "Point Earned": 464}'
```

Respuesta (champion versión 1, gradient boosting):

```json
{"probabilidad_churn": 0.2991, "prediccion": 0, "version_modelo": "1"}
```

Con `metricas.umbral: 0.5` este cliente no se marca como churn. Si se envía un cliente con campos
faltantes, la API responde `422` indicando cuáles.

## 5. Pruebas

```bash
poetry run pytest -q
```
