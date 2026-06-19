from pyspark import pipelines as dp
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType, DoubleType


@dp.materialized_view(
    name="silver_churn",
    comment="Datos limpios, tipados, sin duplicados, con features de negocio derivadas."
)
@dp.expect_all_or_drop({
    "churn_valido": "Churn IN (0, 1)",
    "age_no_nulo": "Age IS NOT NULL",
    "tenure_no_nulo": "Tenure IS NOT NULL",
    "support_calls_no_nulo": "support_calls IS NOT NULL",
    "payment_delay_no_nulo": "payment_delay IS NOT NULL",
    "total_spend_no_nulo": "total_spend IS NOT NULL",
    "age_no_negativa": "Age >= 0",
    "tenure_no_negativa": "Tenure >= 0"
})
def silver_churn():

    df = (
        dp.read("bronze_churn")

        .withColumnRenamed("Usage Frequency", "usage_frequency")
        .withColumnRenamed("Support Calls", "support_calls")
        .withColumnRenamed("Payment Delay", "payment_delay")
        .withColumnRenamed("Subscription Type", "subscription_type")
        .withColumnRenamed("Contract Length", "contract_length")
        .withColumnRenamed("Total Spend", "total_spend")
        .withColumnRenamed("Last Interaction", "last_interaction")

        .drop("CustomerID", "_source_file", "_ingestion_time")

        .withColumn("Age", F.col("Age").cast(IntegerType()))
        .withColumn("Tenure", F.col("Tenure").cast(IntegerType()))
        .withColumn("usage_frequency", F.col("usage_frequency").cast(IntegerType()))
        .withColumn("support_calls", F.col("support_calls").cast(IntegerType()))
        .withColumn("payment_delay", F.col("payment_delay").cast(IntegerType()))
        .withColumn("last_interaction", F.col("last_interaction").cast(IntegerType()))
        .withColumn("total_spend", F.col("total_spend").cast(DoubleType()))
        .withColumn("Churn", F.col("Churn").cast(IntegerType()))

        .dropDuplicates()
    )

    df = (
        df
        .withColumn(
            "support_calls_per_tenure",
            F.when(
                F.col("Tenure") > 0,
                F.col("support_calls") / F.col("Tenure")
            ).otherwise(F.col("support_calls").cast(DoubleType()))
        )
        .withColumn(
            "spend_per_tenure",
            F.when(
                F.col("Tenure") > 0,
                F.col("total_spend") / F.col("Tenure")
            ).otherwise(F.col("total_spend"))
        )
        .withColumn(
            "is_new_client",
            (F.col("Tenure") <= 6).cast(IntegerType())
        )
        .withColumn(
            "high_friction",
            (F.col("support_calls") >= 4).cast(IntegerType())
        )
        .withColumn(
            "high_payment_delay",
            (F.col("payment_delay") >= 15).cast(IntegerType())
        )
        .withColumn(
            "is_inactive",
            (F.col("last_interaction") > 20).cast(IntegerType())
        )
        .withColumn(
            "risk_score_flags",
            F.col("is_new_client")
            + F.col("high_friction")
            + F.col("high_payment_delay")
            + F.col("is_inactive")
        )
    )

    return df