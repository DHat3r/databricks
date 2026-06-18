# Databricks notebook source
# MAGIC %md
# MAGIC # PIPELINE · Capa 🥈 SILVER — Cleansed & Curated
# MAGIC **Banco Futura · Churn Scoring Service**
# MAGIC
# MAGIC Limpieza, tipado, eliminación de duplicados y **feature engineering de negocio**.
# MAGIC Las validaciones de calidad de datos (antes `assert` en el notebook original) ahora son
# MAGIC **expectativas nativas de Lakeflow** (`@dp.expect_all_or_drop`) — si una fila no cumple,
# MAGIC la pipeline la descarta automáticamente y lo reporta en las métricas de calidad, en vez de
# MAGIC abortar toda la ejecución con un `assert`.

# COMMAND ----------

from pyspark import pipelines as dp
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType, DoubleType

# COMMAND ----------

@dp.materialized_view(
    name="silver_churn",
    comment="Datos limpios, tipados, sin duplicados, con features de negocio derivadas."
)
@dp.expect_all_or_drop({
    "churn_valido":              "Churn IN (0, 1)",
    "age_no_nulo":                "Age IS NOT NULL",
    "tenure_no_nulo":              "Tenure IS NOT NULL",
    "support_calls_no_nulo":       "`Support Calls` IS NOT NULL",
    "payment_delay_no_nulo":       "`Payment Delay` IS NOT NULL",
    "total_spend_no_nulo":         "`Total Spend` IS NOT NULL",
    "age_no_negativa":             "Age >= 0",
    "tenure_no_negativa":          "Tenure >= 0",
})
def silver_churn():
    df = (
        dp.read("bronze_churn")
        # 1. Eliminar columnas técnicas de ingesta y el ID de cliente
        .drop("CustomerID", "_source_file", "_ingestion_time")
        # 2. Castear tipos correctos
        .withColumn("Age",              F.col("Age").cast(IntegerType()))
        .withColumn("Tenure",           F.col("Tenure").cast(IntegerType()))
        .withColumn("Usage Frequency",  F.col("Usage Frequency").cast(IntegerType()))
        .withColumn("Support Calls",    F.col("Support Calls").cast(IntegerType()))
        .withColumn("Payment Delay",    F.col("Payment Delay").cast(IntegerType()))
        .withColumn("Last Interaction", F.col("Last Interaction").cast(IntegerType()))
        .withColumn("Total Spend",      F.col("Total Spend").cast(DoubleType()))
        .withColumn("Churn",            F.col("Churn").cast(IntegerType()))
        # 3. Eliminar duplicados exactos (el filtro de target nulo lo hace la expectativa de arriba)
        .dropDuplicates()
    )

    # ---- Feature engineering de negocio ----
    df = (
        df
        # Ratio de soporte por mes de permanencia (fricción relativa)
        .withColumn(
            "support_calls_per_tenure",
            F.when(F.col("Tenure") > 0, F.col("Support Calls") / F.col("Tenure"))
             .otherwise(F.col("Support Calls").cast(DoubleType()))
        )
        # Gasto mensual promedio
        .withColumn(
            "spend_per_tenure",
            F.when(F.col("Tenure") > 0, F.col("Total Spend") / F.col("Tenure"))
             .otherwise(F.col("Total Spend"))
        )
        # Flag: cliente nuevo (primeros 6 meses = período crítico)
        .withColumn("is_new_client", (F.col("Tenure") <= 6).cast(IntegerType()))
        # Flag: alta fricción (≥4 llamadas al soporte)
        .withColumn("high_friction", (F.col("Support Calls") >= 4).cast(IntegerType()))
        # Flag: pago con retraso significativo (≥15 días)
        .withColumn("high_payment_delay", (F.col("Payment Delay") >= 15).cast(IntegerType()))
        # Flag: cliente inactivo (>20 días sin interacción)
        .withColumn("is_inactive", (F.col("Last Interaction") > 20).cast(IntegerType()))
        # Score de riesgo simple (suma de flags)
        .withColumn(
            "risk_score_flags",
            F.col("is_new_client") + F.col("high_friction")
            + F.col("high_payment_delay") + F.col("is_inactive")
        )
    )

    return df
