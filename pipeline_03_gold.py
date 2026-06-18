# Databricks notebook source
# MAGIC %md
# MAGIC # PIPELINE · Capa 🥇 GOLD — Feature Store (ML-ready)
# MAGIC **Banco Futura · Churn Scoring Service**
# MAGIC
# MAGIC Selecciona únicamente las columnas que los modelos necesitan (features + target).
# MAGIC Esta tabla `gold_churn_features` es el **contrato único** que consumen, de forma
# MAGIC independiente, los dos notebooks de modelado (`model_A_xgboost.py` y
# MAGIC `model_B_spark_mllib_nativo.py`).
# MAGIC
# MAGIC Nota: las tablas de **scores** (predicciones de cada modelo) NO se generan aquí —
# MAGIC cada modelo escribe su propia tabla Gold de scores (`gold_churn_scores_xgboost`,
# MAGIC `gold_churn_scores_mllib`) en su propio notebook, porque dependen del modelo entrenado,
# MAGIC que todavía no existe en esta etapa de la pipeline.

# COMMAND ----------

from pyspark import pipelines as dp

TARGET = "Churn"

NUM_FEATURES = [
    "Age", "Tenure", "Usage Frequency",
    "Support Calls", "Payment Delay",
    "Total Spend", "Last Interaction",
    # Nuevas features Silver (feature engineering)
    "support_calls_per_tenure",
    "spend_per_tenure",
    "risk_score_flags",
]

CAT_FEATURES = ["Gender", "Subscription Type", "Contract Length"]

ALL_FEATURES = NUM_FEATURES + CAT_FEATURES

# COMMAND ----------

@dp.materialized_view(
    name="gold_churn_features",
    comment="Tabla ML-ready: features numéricas + categóricas + target, lista para entrenar modelos."
)
def gold_churn_features():
    return dp.read("silver_churn").select(*ALL_FEATURES, TARGET)
