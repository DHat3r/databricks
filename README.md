# Banco Futura · Churn Scoring Service — Databricks Free Edition

Adaptación del notebook original (`ChurnXGBoost_BancoFutura_Spark_Medallion_final.ipynb`) a una
arquitectura nativa de Databricks: **1 Pipeline** (medallion Bronze→Silver→Gold) + **2 modelos
independientes** (XGBoost y Spark MLlib nativo) orquestados por **1 Job**.

```
00_setup_catalogo_volumen.py        ← ejecutar UNA VEZ a mano (crea catálogo/esquema/volumen)

pipeline/
  pipeline_01_bronze.py             ← 3 archivos fuente de UNA SOLA Lakeflow Pipeline
  pipeline_02_silver.py
  pipeline_03_gold.py

models/
  model_A_xgboost.py                ← notebook independiente #1 (sklearn + XGBoost + SHAP)
  model_B_spark_mllib_nativo.py     ← notebook independiente #2 (100% Spark MLlib, sin libs externas)

advanced_asset_bundle/              ← OPCIONAL: si usas Databricks CLI / Asset Bundles
  databricks.yml
  resources/job.yml
```

## Por qué esta arquitectura (y qué cambié respecto al original)

- **Free Edition solo tiene compute serverless, sin GPU**, así que se eliminó toda la lógica de
  detección de GPU (`nvidia-smi`, `device='cuda'`) — todo corre en CPU.
- **La salida a internet en Free Edition está restringida a un set fijo de dominios** y no es
  configurable (eso es función de planes Premium/Enterprise). Por eso `kagglehub` descargando en
  vivo es poco confiable ahí. Se reemplazó por: subir los 2 CSV una vez a un **Volumen de Unity
  Catalog**, y leerlos desde ahí con Spark — cero dependencia de red en cada ejecución.
- Los `assert` de validación de datos del Silver original se convirtieron en **expectativas
  nativas de Lakeflow** (`@dp.expect_all_or_drop`) — la pipeline reporta filas descartadas en sus
  métricas de calidad en vez de abortar con una excepción.
- Se separó la capa **Gold** en dos partes: la tabla de *features* (`gold_churn_features`) la
  genera la Pipeline (es pura transformación, no depende de ningún modelo); las tablas de
  **scores** (`gold_churn_scores_xgboost`, `gold_churn_scores_mllib`) las genera cada notebook de
  modelo por separado, porque necesitan el modelo ya entrenado.
- **Modelo A (XGBoost)**: scikit-learn + `XGBClassifier` + `RandomizedSearchCV` + SHAP, igual que
  el original pero sin GPU y como notebook independiente que lee `gold_churn_features`.
- **Modelo B (Spark MLlib nativo)**: `GBTClassifier` — el árbol de gradient boosting nativo de
  Spark/Databricks, entrenado 100% distribuido sin pasar por pandas ni instalar librerías
  externas. Sirve de punto de comparación frente a XGBoost.

## Paso 1 — Subir el dataset

1. Descarga los 2 CSV desde Kaggle (`muhammadshahidazeem/customer-churn-dataset`) a tu computador:
   `customer_churn_dataset-training-master.csv` y `customer_churn_dataset-testing-master.csv`.
2. En tu workspace de Databricks Free Edition, importa y ejecuta `00_setup_catalogo_volumen.py`
   (Workspace → Import → sube el archivo `.py`, ábrelo, **Run All**). Esto crea el catálogo
   `churn_banco_futura`, el esquema `medallion` y el volumen `raw_data`.
3. Ve a **Catalog Explorer** → `churn_banco_futura` → `medallion` → `raw_data` → botón
   **Upload to this volume** → sube los 2 CSV.

## Paso 2 — Crear la Pipeline (Lakeflow)

1. Importa los 3 archivos de la carpeta `pipeline/` a tu workspace (mantén el mismo orden de
   nombre: `pipeline_01_bronze.py`, `pipeline_02_silver.py`, `pipeline_03_gold.py`).
2. En el menú lateral: **Jobs & Pipelines → Create → ETL Pipeline**.
3. En **Source code**, agrega los 3 archivos como *source files* de la misma pipeline.
4. En **Destination**, configura: Catalog = `churn_banco_futura`, Schema = `medallion`.
5. Compute: deja **Serverless** (única opción en Free Edition).
6. Guarda y haz clic en **Start** / **Run pipeline**. Verifica en el grafo que se crean
   `bronze_churn → silver_churn → gold_churn_features` y revisa la pestaña de calidad de datos
   para ver cuántas filas descartaron las expectativas.

## Paso 3 — Importar los 2 notebooks de modelos

Importa `models/model_A_xgboost.py` y `models/model_B_spark_mllib_nativo.py` a tu workspace.
Puedes correrlos sueltos primero (Run All) para validar que funcionan antes de meterlos al Job —
ambos tienen *widgets* (`catalog`, `schema`) con valores por defecto que coinciden con el Paso 1.

## Paso 4 — Crear el Job que une todo

1. **Jobs & Pipelines → Create → Job**.
2. **Task 1** `run_medallion_pipeline`: tipo **Pipeline** → selecciona la pipeline creada en el
   Paso 2.
3. **Task 2** `train_xgboost`: tipo **Notebook** → `model_A_xgboost.py` → **Depends on**:
   `run_medallion_pipeline`.
4. **Task 3** `train_mllib_native`: tipo **Notebook** → `model_B_spark_mllib_nativo.py` →
   **Depends on**: `run_medallion_pipeline` (queda en paralelo con la Task 2, ambas dependen solo
   de la pipeline).
5. Compute de las tasks de notebook: **Serverless**.
6. Guarda y haz clic en **Run now**. El grafo debería verse así:

```
                 ┌─► train_xgboost          (independiente)
run_medallion_pipeline ─┤
                 └─► train_mllib_native      (independiente)
```

Free Edition permite hasta 5 *job tasks* concurrentes por cuenta — este Job usa 3, así que no hay
problema en correr ambas tasks de modelo en paralelo.

## Resultado final

Tablas en Unity Catalog (`churn_banco_futura.medallion.*`):
- `bronze_churn`, `silver_churn`, `gold_churn_features` — generadas por la Pipeline.
- `gold_churn_scores_xgboost` — generada por el Modelo A, con `prob_churn` y `nivel_riesgo`.
- `gold_churn_scores_mllib` — generada por el Modelo B, mismo formato.

Dos *runs* de MLflow (uno por modelo) con métricas, parámetros y el modelo serializado, listos
para comparar AUC/F1/ROI entre XGBoost y el GBTClassifier nativo, y para promover a Model Registry
si decides ponerlos en producción.

## Opcional — Despliegue por código (Databricks CLI / Asset Bundles)

Si prefieres no usar la UI, la carpeta `advanced_asset_bundle/` trae el mismo Job + Pipeline como
código (`databricks.yml` + `resources/job.yml`). Con la Databricks CLI configurada contra tu
workspace de Free Edition:

```bash
databricks bundle deploy
databricks bundle run churn_job
```

Esto crea la Pipeline y el Job automáticamente, sin pasar por los Pasos 2 y 4 de la UI.
