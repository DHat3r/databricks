# Databricks notebook source
# MAGIC %md
# MAGIC # MODELO B · Spark MLlib nativo (`GBTClassifier`) — Predicción de Churn
# MAGIC **Banco Futura · Churn Scoring Service · Databricks Free Edition**
# MAGIC
# MAGIC Notebook **independiente** (no pertenece a la Pipeline, ni depende de `model_A_xgboost.py`).
# MAGIC Se ejecuta como otra *task* de tipo *Notebook* dentro del **Job**, en paralelo al modelo A,
# MAGIC ambas después de la Pipeline.
# MAGIC
# MAGIC A diferencia del Modelo A, este usa **100% Spark MLlib** (motor nativo de Databricks):
# MAGIC - Sin pasar a pandas: entrena distribuido sobre el `DataFrame` de Spark.
# MAGIC - Sin librerías externas (`xgboost`, `shap`) — todo viene incluido en el runtime.
# MAGIC - `GBTClassifier` (Gradient-Boosted Trees) es el equivalente nativo de Spark a XGBoost.
# MAGIC - Búsqueda de hiperparámetros con `CrossValidator` + `ParamGridBuilder` (el "RandomizedSearchCV"
# MAGIC   de Spark).
# MAGIC
# MAGIC Esto sirve como punto de comparación: ¿vale la pena la complejidad de XGBoost + sklearn,
# MAGIC o un modelo nativo de Spark da resultados comparables con menos piezas móviles?

# COMMAND ----------

dbutils.widgets.text("catalog", "churn_banco_futura", "Catálogo UC")
dbutils.widgets.text("schema", "medallion", "Esquema UC")
dbutils.widgets.text("experiment_path", "/Shared/banco_futura_churn_mllib", "Ruta del experimento MLflow")

CATALOG = dbutils.widgets.get("catalog")
SCHEMA  = dbutils.widgets.get("schema")
EXPERIMENT_PATH = dbutils.widgets.get("experiment_path")
GOLD_TABLE   = f"{CATALOG}.{SCHEMA}.gold_churn_features"
SCORES_TABLE = f"{CATALOG}.{SCHEMA}.gold_churn_scores_mllib"

RANDOM_STATE = 42

# COMMAND ----------

from datetime import datetime
import mlflow
import mlflow.spark
from pyspark.sql import functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

print("✓ Librerías listas (todas nativas de Spark)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Leer la capa Gold (generada por la Pipeline) — como DataFrame de Spark

# COMMAND ----------

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

gold_df = spark.table(GOLD_TABLE)
print(f"✓ [GOLD] Registros: {gold_df.count():,}")
gold_df.groupBy(TARGET).count().orderBy(TARGET).show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Split estratificado 80/20 (nativo en Spark: split por clase y unión)

# COMMAND ----------

train_pos, test_pos = (
    gold_df.filter(F.col(TARGET) == 1)
    .randomSplit([0.8, 0.2], seed=RANDOM_STATE)
)

train_neg, test_neg = (
    gold_df.filter(F.col(TARGET) == 0)
    .randomSplit([0.8, 0.2], seed=RANDOM_STATE)
)

train_df = train_pos.unionByName(train_neg)
test_df  = test_pos.unionByName(test_neg)

n_train = train_df.count()
n_test  = test_df.count()

print(f"✓ Entrenamiento: {n_train:,} | Prueba: {n_test:,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Desbalance de clases — columna de pesos (equivalente a `scale_pos_weight`)

# COMMAND ----------

n_negative = train_df.filter(F.col(TARGET) == 0).count()
n_positive = train_df.filter(F.col(TARGET) == 1).count()
class_weight_ratio = n_negative / n_positive

print(f"  Clase 0 (permanece): {n_negative:,}  |  Clase 1 (churn): {n_positive:,}")
print(f"  Ratio de desbalance (peso clase 1): {class_weight_ratio:.4f}")

train_df = train_df.withColumn(
    "classWeight",
    F.when(F.col(TARGET) == 1, F.lit(class_weight_ratio)).otherwise(F.lit(1.0))
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Pipeline de ML nativo de Spark
# MAGIC `StringIndexer` → `OneHotEncoder` para categóricas, `VectorAssembler` combina todo,
# MAGIC `GBTClassifier` como modelo (usa `weightCol` para compensar el desbalance).

# COMMAND ----------

indexers = [
    StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep")
    for c in CAT_FEATURES
]
encoder = OneHotEncoder(
    inputCols=[f"{c}_idx" for c in CAT_FEATURES],
    outputCols=[f"{c}_ohe" for c in CAT_FEATURES],
)
assembler = VectorAssembler(
    inputCols=NUM_FEATURES + [f"{c}_ohe" for c in CAT_FEATURES],
    outputCol="features",
    handleInvalid="keep",
)

gbt = GBTClassifier(
    labelCol=TARGET,
    featuresCol="features",
    weightCol="classWeight",
    seed=RANDOM_STATE,
)

ml_pipeline = Pipeline(stages=indexers + [encoder, assembler, gbt])

print("✓ Pipeline Spark ML definido: StringIndexer → OneHotEncoder → VectorAssembler → GBTClassifier")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Búsqueda de hiperparámetros — `CrossValidator`
# MAGIC Equivalente nativo de Spark a `RandomizedSearchCV`. Rejilla moderada (8 combinaciones × 3
# MAGIC folds = 24 entrenamientos) para que sea liviano en compute serverless.

# COMMAND ----------


import os
import gc
from datetime import datetime

# ============================================================
# FIX Databricks Serverless / Shared (OBLIGATORIO)
# ============================================================

volume_path = f"{CATALOG}.{SCHEMA}.sparkml_tmp"
spark.sql(f"CREATE VOLUME IF NOT EXISTS {volume_path}")

os.environ["SPARKML_TEMP_DFS_PATH"] = f"/Volumes/{CATALOG}/{SCHEMA}/sparkml_tmp"

print("✓ SparkML temp path listo")


# ============================================================
# FIX CV (IMPORTANTE)
# ============================================================

param_grid = (
    ParamGridBuilder()
    .addGrid(gbt.maxDepth, [3, 5])
    .addGrid(gbt.maxIter, [100])
    .addGrid(gbt.stepSize, [0.1])
    .build()
)

cv = CrossValidator(
    estimator=ml_pipeline,
    estimatorParamMaps=param_grid,
    evaluator=evaluator_auc,
    numFolds=3,
    seed=RANDOM_STATE,
    parallelism=1,
    collectSubModels=False
)

print("Iniciando CV (2 × 3 = 6 entrenamientos)...")

start_time = datetime.now()
cv_model = cv.fit(train_df)
elapsed = (datetime.now() - start_time).seconds

best_model = cv_model.bestModel
best_gbt = best_model.stages[-1]

print(f"✓ Búsqueda completada en {elapsed}s | Mejor AUC-ROC (CV): {best_auc_cv:.4f}")
print(f"  maxDepth={best_gbt.getMaxDepth()}  maxIter={best_gbt.getMaxIter()}  stepSize={best_gbt.getStepSize()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Evaluación sobre el conjunto de prueba

# COMMAND ----------

predictions = best_model.transform(test_df)

extract_prob_udf = F.udf(lambda v: float(v[1]), "double")
predictions = predictions.withColumn("prob_churn", extract_prob_udf(F.col("probability")))

auc = evaluator_auc.evaluate(predictions)

eval_acc  = MulticlassClassificationEvaluator(labelCol=TARGET, predictionCol="prediction", metricName="accuracy")
eval_f1   = MulticlassClassificationEvaluator(labelCol=TARGET, predictionCol="prediction", metricName="f1")
eval_prec = MulticlassClassificationEvaluator(labelCol=TARGET, predictionCol="prediction", metricName="weightedPrecision")
eval_rec  = MulticlassClassificationEvaluator(labelCol=TARGET, predictionCol="prediction", metricName="weightedRecall")

acc  = eval_acc.evaluate(predictions)
f1   = eval_f1.evaluate(predictions)
prec = eval_prec.evaluate(predictions)
rec  = eval_rec.evaluate(predictions)

print("=" * 50)
print("  MÉTRICAS DE EVALUACIÓN — GBTClassifier (Spark nativo)")
print("=" * 50)
print(f"  Accuracy  : {acc:.4f}")
print(f"  Precision : {prec:.4f}")
print(f"  Recall    : {rec:.4f}")
print(f"  F1 Score  : {f1:.4f}")
print(f"  ROC-AUC   : {auc:.4f}")

print("\nMatriz de confusión:")
predictions.groupBy(TARGET, "prediction").count().orderBy(TARGET, "prediction").show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Importancia de variables (nativa de Spark, sin SHAP)

# COMMAND ----------

assembler_stage = best_model.stages[-2]
feature_names = assembler_stage.getInputCols()
importances = best_gbt.featureImportances.toArray()

importance_df = (
    spark.createDataFrame(
        list(zip(feature_names, [float(i) for i in importances])),
        ["feature", "importance"],
    )
    .orderBy(F.desc("importance"))
)

print("Top variables por importancia (GBTClassifier.featureImportances):")
importance_df.show(10, truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Registro en MLflow

# COMMAND ----------

mlflow.set_experiment(EXPERIMENT_PATH)

with mlflow.start_run(run_name="gbt_churn_banco_futura") as run:
    mlflow.log_param("maxDepth", best_gbt.getMaxDepth())
    mlflow.log_param("maxIter", best_gbt.getMaxIter())
    mlflow.log_param("stepSize", best_gbt.getStepSize())
    mlflow.log_param("class_weight_ratio", class_weight_ratio)
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", prec)
    mlflow.log_metric("recall", rec)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("roc_auc", auc)
    mlflow.spark.log_model(best_model, artifact_path="model")
    run_id = run.info.run_id

print(f"✓ Run registrado en MLflow: {run_id}")

# Opcional: registrar el modelo en el Model Registry de Unity Catalog.
# mlflow.set_registry_uri("databricks-uc")
# mlflow.register_model(f"runs:/{run_id}/model", f"{CATALOG}.{SCHEMA}.churn_gbt_model")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Scoring masivo sobre toda la cartera → tabla Gold de scores

# COMMAND ----------

full_scores = best_model.transform(gold_df).withColumn("prob_churn", extract_prob_udf(F.col("probability")))

gold_scores_df = (
    full_scores
    .withColumn(
        "nivel_riesgo",
        F.when(F.col("prob_churn") <= 0.40, "BAJO")
         .when(F.col("prob_churn") <= 0.70, "MEDIO")
         .otherwise("ALTO")
    )
    .withColumn("modelo", F.lit("spark_mllib_gbt"))
    .withColumn("scoring_timestamp", F.current_timestamp())
    .withColumn("mlflow_run_id", F.lit(run_id))
    .select(*NUM_FEATURES, *CAT_FEATURES, TARGET, "prob_churn", "nivel_riesgo",
            "modelo", "scoring_timestamp", "mlflow_run_id")
)

gold_scores_df.write.format("delta").mode("overwrite").saveAsTable(SCORES_TABLE)

print(f"✓ Tabla de scores escrita en: {SCORES_TABLE}")
gold_scores_df.groupBy("nivel_riesgo").count().orderBy("nivel_riesgo").show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Impacto financiero estimado

# COMMAND ----------

alto_riesgo = gold_scores_df.filter(F.col("nivel_riesgo") == "ALTO").count()
total_clientes = gold_scores_df.count()
valor_dolar    = 890
CLV_PROMEDIO   = 1_200 * valor_dolar
TASA_RETENCION = 0.25
COSTO_CAMPANA  = 50_000 * valor_dolar

clientes_retenidos = int(alto_riesgo * TASA_RETENCION)
ingreso_preservado = clientes_retenidos * CLV_PROMEDIO
roi = ingreso_preservado / COSTO_CAMPANA

print("=" * 55)
print("  IMPACTO FINANCIERO ESTIMADO — Modelo GBTClassifier (Spark nativo)")
print("=" * 55)
print(f"  Clientes analizados   : {total_clientes:,}")
print(f"  Clientes ALTO RIESGO  : {alto_riesgo:,}")
print(f"  Clientes retenidos    : {clientes_retenidos:,}")
print(f"  Ingreso preservado    : ${ingreso_preservado:,}")
print(f"  Costo campaña         : ${COSTO_CAMPANA:,}")
print(f"  ROI estimado          : {roi:.1f}x")

mlflow.start_run(run_id=run_id)
mlflow.log_metric("roi_estimado", roi)
mlflow.log_metric("clientes_alto_riesgo", float(alto_riesgo))
mlflow.end_run()
