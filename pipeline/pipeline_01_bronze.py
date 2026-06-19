from pyspark import pipelines as dp
from pyspark.sql import functions as F

RAW_VOLUME_PATH = "/Volumes/churn_banco_futura/medallion/raw_data"

TRAIN_FILE = "customer_churn_dataset-training-master.csv"
TEST_FILE = "customer_churn_dataset-testing-master.csv"


@dp.materialized_view(
    name="bronze_churn",
    comment="Datos crudos de churn (train+test) con nombres de columnas estandarizados."
)
def bronze_churn():

    def read_file(file_name):
        return (
            spark.read.format("csv")
            .option("header", "true")
            .option("inferSchema", "true")
            .load(f"{RAW_VOLUME_PATH}/{file_name}")

            .withColumnRenamed("Usage Frequency", "usage_frequency")
            .withColumnRenamed("Support Calls", "support_calls")
            .withColumnRenamed("Payment Delay", "payment_delay")
            .withColumnRenamed("Subscription Type", "subscription_type")
            .withColumnRenamed("Contract Length", "contract_length")
            .withColumnRenamed("Total Spend", "total_spend")
            .withColumnRenamed("Last Interaction", "last_interaction")

            .withColumn("_source_file", F.lit(file_name))
        )

    train_df = read_file(TRAIN_FILE)
    test_df = read_file(TEST_FILE)

    return (
        train_df.unionByName(test_df)
        .withColumn("_ingestion_time", F.current_timestamp())
    )