# Actividad 3: drift y tests

## Qué se añadió

```
configs/config.yaml          secciones drift (nivel, candidatas...) y monitoreo (umbral, alfa)
domain/drift.py              emulación y detección de drift (sin MLflow ni Prefect)
domain/modelo.py             + dividir() y predecir(), compartidas por flows, API y tests
scripts/lotes.py             lote más reciente y datos de referencia (entrenamiento)
flows/inferencia_flow.py     ahora puntúa lotes de la parte de prueba con drift emulado
flows/drift_features_flow.py drift de cada variable con Evidently, reporte HTML por lote
flows/drift_evento_flow.py   drift de la tasa de churn (chi-cuadrado), exige etiquetas
flows/desplegar.py           + despliegues de los dos pipelines de drift
api/main.py                  carga del champion en cargar_champion() (sustituible en tests)
tests/                       datos, modelo, inferencia, API (modelo falso) y drift
.github/workflows/tests.yml  CI: pytest en cada Pull Request
```

## Cómo se emula el drift

El nivel se fija en `configs/config.yaml` (`drift.nivel`), no al ejecutar. Con un nivel entre 0.0 y 1.0:

1. **Target.** El lote se muestrea de la **parte de prueba** de forma estratificada para que la tasa de churn sea
   `tasa_base + nivel × subida_tasa_max`. Las etiquetas siguen siendo reales (no se invierten).
2. **Variables.** Se desplazan las primeras `round(nivel × 8)` columnas de `drift.candidatas`, en orden:
   - numéricas: `+ nivel × 1.5` desviaciones estándar;
   - categóricas: una fracción `nivel` de las filas pasa a la categoría de `categoria_destino`
     (Geography → Germany, Card Type → DIAMOND).

| Nivel | Columnas desplazadas | Desplazamiento numérico | Tasa de churn |
|---|---|---|---|
| 0.0 | ninguna | — | 20,4 % (sin cambio) |
| 0.3 | Age, Balance | 0,45 desv. estándar | ≈ 29 % |
| 0.6 | Age, Balance, Geography, CreditScore, EstimatedSalary | 0,9 desv. estándar | ≈ 38 % |

Con `drift.semilla: 42`, el mismo nivel produce siempre el mismo lote.

Cada inferencia guarda una copia del lote puntuado, **con sus etiquetas reales**, en
`data/lotes/lote_<fecha>_<hora>_nivel<n>.csv`. Las predicciones van a `data/predicciones/` sin la etiqueta.

## Detección

- **Features** (`drift_features_flow.py`): Evidently `DataDriftPreset` compara las 13 variables del modelo del lote con
  la parte de entrenamiento (Wasserstein para numéricas, Jensen-Shannon para categóricas, umbral 0,1). Guarda
  `reports/drift/drift_features_<lote>.html` y un `.json` con el resumen. **Alerta** si la fracción de variables con
  drift supera `monitoreo.umbral_fraccion_drift` (0,3).
- **Evento** (`drift_evento_flow.py`): chi-cuadrado de la tasa de `Exited` del lote frente a la de entrenamiento. Alerta
  si el p-valor < `monitoreo.alfa_evento` (0,05). Si el lote no trae etiquetas o le faltan, el flow **falla con un
  mensaje claro**.

Las alertas aparecen como `WARNING` en los logs de Prefect, como tag `alerta` en el run de MLflow y en un artefacto
Markdown de Prefect.

## Resultados

| Nivel | Variables con drift | Fracción | Alerta features | Tasa lote | Alerta evento |
|---|---|---|---|---|---|
| 0.0 | 1 (CreditScore, por azar del muestreo) | 0,08 | no | 20,3 % | no (p = 1,0) |
| 0.3 | 3 | 0,23 | no | 29,3 % | sí (p = 0,0002) |
| 0.6 | 5 | 0,38 | **sí** | 38,3 % | sí (p < 0,0001) |

## Cómo ejecutarlo

Con las variables de entorno definidas (`PREFECT_API_URL`, `MLFLOW_TRACKING_URI`) y un champion registrado:

```bash
# Repetir para nivel 0.0, 0.3 y 0.6 (cambiar drift.nivel en configs/config.yaml)
poetry run python flows/inferencia_flow.py
poetry run python flows/drift_features_flow.py   # usa el último lote
poetry run python flows/drift_evento_flow.py
```

Para analizar un lote concreto: `poetry run python flows/drift_features_flow.py data/lotes/<archivo>.csv`.

## Tests

```bash
poetry run pytest
```

| Archivo | Qué comprueba |
|---|---|
| `test_datos.py` | esquema, tipos, rangos, binarias, categorías, nulos, duplicados, validación de carga |
| `test_modelo.py` | selección ALL/NUMERIC/CUSTOM, división estratificada, pipeline (AUC > 0,7), métricas |
| `test_inferencia.py` | predicción y umbral con modelo falso, búsqueda del último lote |
| `test_api.py` | `/health`, `/predict` y errores 422 con un modelo falso, sin MLflow |
| `test_drift.py` | emulación (tasa, columnas, semilla), etiquetas, chi-cuadrado, alerta y que solo el 0.6 supere el umbral con Evidently |

El workflow `.github/workflows/tests.yml` los ejecuta en GitHub en cada Pull Request.
