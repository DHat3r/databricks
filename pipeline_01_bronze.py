# Databricks notebook source
# MAGIC %md
# MAGIC # PIPELINE · Capa 🥉 BRONZE — Raw Ingestion
# MAGIC **Banco Futura · Churn Scoring Service**
# MAGIC
# MAGIC Uno de los 3 archivos fuente de la **Lakeflow Spark Declarative Pipeline**
# MAGIC (los otros dos son `pipeline_02_silver.py` y `pipeline_03_gold.py`).
# MAGIC Los tres en conjunto forman **una sola Pipeline** en Databricks.
# MAGIC
# MAGIC **Principios Bronze:** sin transformaciones, sin filtros, dato tal cual llega.
# MAGIC Lee los 2 CSV (train + test) directamente desde el Volumen de Unity Catalog
# MAGIC (ver `00_setup_catalogo_volumen.py`), sin depender de internet ni de la API de Kaggle.

# COMMAND ----------

from pyspark import pipelines as dp
from pyspark.sql import functions as F

# ⚠️ EDITA ESTA RUTA si usaste otro catálogo/esquema/volumen en el setup
RAW_VOLUME_PATH = "/Volumes/churn_banco_futura/medallion/raw_data"

TRAIN_FILE = "customer_churn_dataset-training-master.csv"
TEST_FILE  = "customer_churn_dataset-testing-master.csv"

# COMMAND ----------

@dp.materialized_view(
    name="bronze_churn",
    comment="Datos crudos de churn (train+test) tal como llegan de Kaggle, sin transformar."
)
def bronze_churn():
    def _read(file_name):
        return (
            spark.read.format("csv")
            .option("header", "true")
            .option("inferSchema", "true")
            .load(f"{RAW_VOLUME_PATH}/{file_name}")
            .withColumn("_source_file", F.lit(file_name))
        )

    train_df = _read(TRAIN_FILE)
    test_df  = _read(TEST_FILE)

    return (
        train_df.unionByName(test_df)
        .withColumn("_ingestion_time", F.current_timestamp())
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ### Nota — Auto Loader para producción
# MAGIC Para un caso real con llegada continua de archivos, esta tabla debería declararse con
# MAGIC `@dp.table()` (streaming table) y leer con `spark.readStream.format("cloudFiles")`
# MAGIC (Auto Loader), para detectar incrementalmente nuevos CSV que lleguen al Volumen.
# MAGIC Para este caso (2 archivos estáticos de un dataset académico) un `materialized_view`
# MAGIC batch es más simple y suficiente.
