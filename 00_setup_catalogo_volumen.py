# Databricks notebook source
# MAGIC %md
# MAGIC # 00 · Setup — Catálogo, Esquema y Volumen (Unity Catalog)
# MAGIC **Banco Futura · Churn Scoring Service · Databricks Free Edition**
# MAGIC
# MAGIC Este notebook se ejecuta **una sola vez, manualmente**, antes de crear la Pipeline y el Job.
# MAGIC No forma parte de la Pipeline ni del Job — es solo preparación de infraestructura.
# MAGIC
# MAGIC Pasos que realiza:
# MAGIC 1. Crea el catálogo y el esquema de Unity Catalog donde vivirán las tablas Bronze/Silver/Gold.
# MAGIC 2. Crea un **Volumen** llamado `raw_data` — ahí subirás manualmente los 2 CSV de Kaggle.
# MAGIC
# MAGIC ### ¿Por qué un Volumen y no `kagglehub` descargando directo?
# MAGIC Databricks Free Edition solo tiene **compute serverless** y el **acceso a internet de salida
# MAGIC está restringido a un set fijo de dominios de confianza** (no es configurable en Free Edition,
# MAGIC eso es una función de los planes Premium/Enterprise). Eso significa que llamadas a la API de
# MAGIC Kaggle pueden fallar de forma intermitente o quedar bloqueadas. La forma robusta y 100% nativa
# MAGIC de Databricks es: descargar el dataset una vez desde tu navegador → subirlo a un Volumen de
# MAGIC Unity Catalog (drag & drop, sin código) → leerlo desde ahí con Spark, sin depender de red externa.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Parámetros (edítalos si quieres otros nombres)

# COMMAND ----------

dbutils.widgets.text("catalog", "churn_banco_futura", "Catálogo UC")
dbutils.widgets.text("schema", "medallion", "Esquema UC")
dbutils.widgets.text("volume", "raw_data", "Volumen (Bronze landing zone)")

CATALOG = dbutils.widgets.get("catalog")
SCHEMA  = dbutils.widgets.get("schema")
VOLUME  = dbutils.widgets.get("volume")

print(f"Catálogo : {CATALOG}")
print(f"Esquema  : {SCHEMA}")
print(f"Volumen  : {VOLUME}")

# COMMAND ----------

spark.sql(f"CREATE CATALOG IF NOT EXISTS {CATALOG}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")
spark.sql(f"CREATE VOLUME IF NOT EXISTS {CATALOG}.{SCHEMA}.{VOLUME}")

volume_path = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}"
print(f"✓ Catálogo, esquema y volumen listos.")
print(f"  Ruta del volumen: {volume_path}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Siguiente paso (manual, fuera de Databricks)
# MAGIC
# MAGIC 1. Descarga los 2 CSV del dataset de Kaggle **customer-churn-dataset**
# MAGIC    (`muhammadshahidazeem/customer-churn-dataset`) a tu computador:
# MAGIC    - `customer_churn_dataset-training-master.csv`
# MAGIC    - `customer_churn_dataset-testing-master.csv`
# MAGIC 2. En el workspace de Databricks, ve a **Catalog Explorer** (ícono de catálogo en la barra
# MAGIC    lateral) → navega hasta `{catalog}.{schema}.{volume}` → botón **Upload to this volume**
# MAGIC    → arrastra los 2 archivos CSV.
# MAGIC 3. Verifica con la celda de abajo que ambos archivos quedaron en el volumen.

# COMMAND ----------

files = dbutils.fs.ls(volume_path)
if not files:
    print("⚠ El volumen está vacío todavía. Sube los 2 CSV antes de crear la Pipeline.")
else:
    print("Archivos encontrados en el volumen:")
    for f in files:
        print(f"  - {f.name}  ({f.size:,} bytes)")
