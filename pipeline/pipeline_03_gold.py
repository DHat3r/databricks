from pyspark import pipelines as dp

TARGET = "Churn"

NUM_FEATURES = [
    "Age",
    "Tenure",
    "usage_frequency",
    "support_calls",
    "payment_delay",
    "total_spend",
    "last_interaction",
    "support_calls_per_tenure",
    "spend_per_tenure",
    "risk_score_flags",
]

CAT_FEATURES = [
    "Gender",
    "subscription_type",
    "contract_length",
]

ALL_FEATURES = NUM_FEATURES + CAT_FEATURES


@dp.materialized_view(
    name="gold_churn_features",
    comment="Tabla ML-ready lista para entrenar modelos."
)
def gold_churn_features():
    return dp.read("silver_churn").select(*ALL_FEATURES, TARGET)