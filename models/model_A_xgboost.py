# Databricks notebook source
# MAGIC %md
# MAGIC # MODELO A · XGBoost — Predicción de Churn
# MAGIC **Banco Futura · Churn Scoring Service · Databricks Free Edition**
# MAGIC
# MAGIC Notebook **independiente** (no pertenece a la Pipeline). Se ejecuta como una *task* de
# MAGIC tipo *Notebook* dentro del **Job**, después de que la Pipeline haya generado
# MAGIC `gold_churn_features`.
# MAGIC
# MAGIC Sigue la misma lógica que el notebook original: preprocesamiento con scikit-learn,
# MAGIC `XGBClassifier` con `RandomizedSearchCV`, manejo de desbalance con `scale_pos_weight`,
# MAGIC evaluación completa, explicabilidad con SHAP y registro en MLflow.
# MAGIC
# MAGIC ⚠️ Free Edition no tiene GPU (solo compute serverless) — todo corre en CPU. El espacio de
# MAGIC búsqueda de hiperparámetros se mantiene moderado para que termine en un tiempo razonable.

# COMMAND ----------

# MAGIC %pip install xgboost shap scikit-learn --quiet
dbutils.library.restartPython()

# COMMAND ----------

dbutils.widgets.text("catalog", "churn_banco_futura", "Catálogo UC")
dbutils.widgets.text("schema", "medallion", "Esquema UC")
dbutils.widgets.text("experiment_path", "/Shared/banco_futura_churn_xgboost", "Ruta del experimento MLflow")

CATALOG = dbutils.widgets.get("catalog")
SCHEMA  = dbutils.widgets.get("schema")
EXPERIMENT_PATH = dbutils.widgets.get("experiment_path")
GOLD_TABLE = f"{CATALOG}.{SCHEMA}.gold_churn_features"
SCORES_TABLE = f"{CATALOG}.{SCHEMA}.gold_churn_scores_xgboost"

RANDOM_STATE = 42

# COMMAND ----------

import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve
)
from xgboost import XGBClassifier

print("✓ Librerías listas")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Leer la capa Gold (generada por la Pipeline) y pasar a pandas
# MAGIC Igual que en el notebook original: Spark gestiona Bronze/Silver/Gold; XGBoost entrena en
# MAGIC pandas (volumen de datos pequeño/mediano — para datasets muy grandes usar XGBoost4J-Spark).

# COMMAND ----------

gold_df = spark.table(GOLD_TABLE)

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

gold_pd = gold_df.toPandas()
X = gold_pd[ALL_FEATURES]
y = gold_pd[TARGET]

print(f"✓ [GOLD] X shape: {X.shape}")
print(f"  Distribución target:\n{y.value_counts().to_string()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Pipeline de preprocesamiento (scikit-learn)

# COMMAND ----------

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler",  StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot",  OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer,     NUM_FEATURES),
        ("cat", categorical_transformer, CAT_FEATURES),
    ],
    remainder="drop",
)

print("✓ Pipeline de preprocesamiento definido")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Split estratificado 80/20

# COMMAND ----------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
)

print(f"✓ Entrenamiento: {len(X_train):,}  |  Prueba: {len(X_test):,}")
print(f"  Tasa churn train: {y_train.mean():.2%}  |  Tasa churn test: {y_test.mean():.2%}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Desbalance de clases — `scale_pos_weight`

# COMMAND ----------

n_negative = (y_train == 0).sum()
n_positive = (y_train == 1).sum()
scale_pos_weight = n_negative / n_positive

print(f"  Clase 0 (permanece): {n_negative:,}  |  Clase 1 (churn): {n_positive:,}")
print(f"  scale_pos_weight: {scale_pos_weight:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Pipeline completo (preprocesamiento + XGBoost) y búsqueda de hiperparámetros

# COMMAND ----------

xgb_base = XGBClassifier(
    objective="binary:logistic",
    scale_pos_weight=scale_pos_weight,
    eval_metric="auc",
    random_state=RANDOM_STATE,
    tree_method="hist",   # rápido en CPU; Free Edition no tiene GPU
    device="cpu",
    n_jobs=-1,
)

pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", xgb_base),
])

param_dist = {
    "model__n_estimators":     [200, 300, 400],
    "model__max_depth":        [3, 4, 5],
    "model__learning_rate":    [0.03, 0.05, 0.08],
    "model__subsample":        [0.7, 0.8, 0.9],
    "model__colsample_bytree": [0.7, 0.8, 0.9],
    "model__reg_alpha":        [0.1, 0.5, 1.0],
    "model__reg_lambda":       [1.0, 2.0, 5.0],
    "model__min_child_weight": [3, 5, 7],
    "model__gamma":            [0.1, 0.2, 0.5],
}

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

random_search = RandomizedSearchCV(
    estimator=pipeline,
    param_distributions=param_dist,
    n_iter=20,             # 20 combinaciones × 3 folds = 60 entrenamientos
    scoring="roc_auc",
    cv=cv,
    n_jobs=-1,
    verbose=1,
    random_state=RANDOM_STATE,
)

print("Iniciando búsqueda de hiperparámetros (20 × 3 = 60 entrenamientos)...")
start_time = datetime.now()
random_search.fit(X_train, y_train)
elapsed = (datetime.now() - start_time).seconds

best_model = random_search.best_estimator_
print(f"✓ Búsqueda completada en {elapsed}s | Mejor AUC-ROC (CV): {random_search.best_score_:.4f}")
for k, v in random_search.best_params_.items():
    print(f"  {k.replace('model__', '')}: {v}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Evaluación sobre el conjunto de prueba

# COMMAND ----------

y_pred      = best_model.predict(X_test)
y_pred_prob = best_model.predict_proba(X_test)[:, 1]

acc  = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec  = recall_score(y_test, y_pred)
f1   = f1_score(y_test, y_pred)
auc  = roc_auc_score(y_test, y_pred_prob)

print("=" * 50)
print("  MÉTRICAS DE EVALUACIÓN — XGBoost")
print("=" * 50)
print(f"  Accuracy  : {acc:.4f}")
print(f"  Precision : {prec:.4f}")
print(f"  Recall    : {rec:.4f}")
print(f"  F1 Score  : {f1:.4f}")
print(f"  ROC-AUC   : {auc:.4f}")
print(classification_report(y_test, y_pred))

# COMMAND ----------

cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
            xticklabels=["Pred: No Churn", "Pred: Churn"],
            yticklabels=["Real: No Churn", "Real: Churn"])
ax.set_title("Matriz de Confusión — XGBoost", fontweight="bold")
plt.tight_layout()
plt.savefig("/tmp/confusion_matrix_xgb.png", dpi=150, bbox_inches="tight")
plt.show()

# COMMAND ----------

fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
youden_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[youden_idx]

fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(fpr, tpr, color="#2196F3", lw=2.5, label=f"XGBoost (AUC = {auc:.4f})")
ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Aleatorio (AUC = 0.50)")
ax.scatter(fpr[youden_idx], tpr[youden_idx], color="#F44336", s=80, zorder=5,
           label=f"Umbral óptimo = {optimal_threshold:.2f}")
ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
ax.set_title("Curva ROC — XGBoost", fontweight="bold")
ax.legend()
plt.tight_layout()
plt.savefig("/tmp/roc_curve_xgb.png", dpi=150, bbox_inches="tight")
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Explicabilidad con SHAP

# COMMAND ----------

preprocessor_fit = best_model.named_steps["preprocessor"]
xgb_model = best_model.named_steps["model"]

cat_names = preprocessor_fit.named_transformers_["cat"].named_steps["onehot"].get_feature_names_out(CAT_FEATURES)
all_feature_names = NUM_FEATURES + list(cat_names)

X_test_processed = preprocessor_fit.transform(X_test)
explainer  = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test_processed)

plt.figure(figsize=(9, 6))
shap.summary_plot(shap_values, X_test_processed, feature_names=all_feature_names,
                   plot_type="dot", max_display=12, show=False)
plt.title("SHAP Summary Plot — XGBoost", fontweight="bold")
plt.tight_layout()
plt.savefig("/tmp/shap_summary_xgb.png", dpi=150, bbox_inches="tight")
plt.show()

shap_importance = pd.DataFrame({
    "feature": all_feature_names,
    "mean_abs_shap": np.abs(shap_values).mean(axis=0),
}).sort_values("mean_abs_shap", ascending=False)

print(shap_importance.head(10).to_string(index=False))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Registro en MLflow

# COMMAND ----------

mlflow.set_experiment(EXPERIMENT_PATH)

with mlflow.start_run(run_name="xgboost_churn_banco_futura") as run:
    mlflow.log_params(random_search.best_params_)
    mlflow.log_param("scale_pos_weight", scale_pos_weight)
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", prec)
    mlflow.log_metric("recall", rec)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("roc_auc", auc)
    mlflow.log_artifact("/tmp/confusion_matrix_xgb.png")
    mlflow.log_artifact("/tmp/roc_curve_xgb.png")
    mlflow.log_artifact("/tmp/shap_summary_xgb.png")
    mlflow.sklearn.log_model(best_model, artifact_path="model", input_example=X_train.head(5))
    run_id = run.info.run_id

print(f"✓ Run registrado en MLflow: {run_id}")

# Opcional: registrar el modelo en el Model Registry de Unity Catalog.
# Requiere permisos CREATE MODEL en el catálogo y descomentar estas líneas:
#
# mlflow.set_registry_uri("databricks-uc")
# mlflow.register_model(f"runs:/{run_id}/model", f"{CATALOG}.{SCHEMA}.churn_xgboost_model")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Scoring masivo sobre toda la cartera → tabla Gold de scores

# COMMAND ----------

X_full_processed = preprocessor_fit.transform(X)
prob_churn_full = xgb_model.predict_proba(X_full_processed)[:, 1]

scores_pd = gold_pd[ALL_FEATURES + [TARGET]].copy()
scores_pd["prob_churn"] = prob_churn_full
scores_pd["nivel_riesgo"] = pd.cut(
    scores_pd["prob_churn"], bins=[-0.001, 0.40, 0.70, 1.001], labels=["BAJO", "MEDIO", "ALTO"]
).astype(str)
scores_pd["modelo"] = "xgboost"
scores_pd["scoring_timestamp"] = datetime.now().isoformat()
scores_pd["mlflow_run_id"] = run_id

gold_scores_df = spark.createDataFrame(scores_pd)
gold_scores_df.write.format("delta").mode("overwrite").saveAsTable(SCORES_TABLE)

print(f"✓ Tabla de scores escrita en: {SCORES_TABLE}")
gold_scores_df.groupBy("nivel_riesgo").count().orderBy("nivel_riesgo").show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Impacto financiero estimado

# COMMAND ----------

alto_riesgo    = (scores_pd["nivel_riesgo"] == "ALTO").sum()
total_clientes = len(scores_pd)
valor_dolar    = 890
CLV_PROMEDIO   = 1_200 * valor_dolar
TASA_RETENCION = 0.25
COSTO_CAMPANA  = 50_000 * valor_dolar

clientes_retenidos = int(alto_riesgo * TASA_RETENCION)
ingreso_preservado = clientes_retenidos * CLV_PROMEDIO
roi = ingreso_preservado / COSTO_CAMPANA

print("=" * 55)
print("  IMPACTO FINANCIERO ESTIMADO — Modelo XGBoost")
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
