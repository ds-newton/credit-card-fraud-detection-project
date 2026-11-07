"""
Credit Card Fraud Detection — Complete Project Script
Dataset: creditcard.csv (ULB / Kaggle)
"""

# ============================================================
# 0. IMPORTS
# ============================================================
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss
from imblearn.pipeline import Pipeline as ImbPipeline

sns.set_style("whitegrid")
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ============================================================
# 1. LOAD DATA & BASIC UNDERSTANDING
# ============================================================
df = pd.read_csv("data/creditcard.csv")

print("Shape:", df.shape)
print("\nMissing values:\n", df.isnull().sum().sum())
print("\nClass distribution:\n", df["Class"].value_counts())
print("\nFraud ratio: {:.4f}%".format(df["Class"].mean() * 100))

# ============================================================
# 2. EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================

# 2.1 Class imbalance visualisation
plt.figure(figsize=(5, 4))
sns.countplot(x="Class", data=df, palette="Set2")
plt.title("Class Distribution (0 = Legit, 1 = Fraud)")
plt.yscale("log")          # log scale to make the minority bar visible
plt.show()

# 2.2 Transaction amount distribution by class
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
sns.histplot(df[df["Class"] == 0]["Amount"], bins=60, ax=axes[0], color="steelblue")
axes[0].set_title("Legitimate – Amount Distribution")
sns.histplot(df[df["Class"] == 1]["Amount"], bins=60, ax=axes[1], color="crimson")
axes[1].set_title("Fraud – Amount Distribution")
plt.tight_layout()
plt.show()

# 2.3 Correlation heatmap (V1–V28 + Amount + Class)
plt.figure(figsize=(18, 12))
corr = df.corr()
sns.heatmap(corr, cmap="coolwarm", center=0, cbar_kws={"shrink": 0.8})
plt.title("Correlation Matrix")
plt.tight_layout()
plt.show()

# 2.4 Boxplots for outlier inspection (Amount and a few PCA components)
features_to_plot = ["Amount", "V1", "V2", "V3", "V7", "V10", "V12", "V14", "V17"]
fig, axes = plt.subplots(3, 3, figsize=(15, 10))
for ax, feat in zip(axes.ravel(), features_to_plot):
    sns.boxplot(x="Class", y=feat, data=df, ax=ax, palette="Set3")
    ax.set_title(feat)
plt.tight_layout()
plt.show()

# 2.5 IQR-based outlier count (Amount)
Q1 = df["Amount"].quantile(0.25)
Q3 = df["Amount"].quantile(0.75)
IQR = Q3 - Q1
lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
outliers = df[(df["Amount"] < lower) | (df["Amount"] > upper)]
print(f"Amount outliers (IQR rule): {len(outliers)}")

# ============================================================
# 3. DATA PREPROCESSING
# ============================================================

# 3.1 No missing values were found earlier; confirm and drop duplicates if any.
print("Duplicates:", df.duplicated().sum())
df = df.drop_duplicates().reset_index(drop=True)
print("Shape after de-duplication:", df.shape)

# 3.2 Feature scaling
#     V1–V28 are already PCA-scaled. We scale Time and Amount.
scaler = StandardScaler()
df["scaled_amount"] = scaler.fit_transform(df["Amount"].values.reshape(-1, 1))
df["scaled_time"]   = scaler.fit_transform(df["Time"].values.reshape(-1, 1))

# Drop the original Time and Amount to avoid redundancy
df_model = df.drop(["Time", "Amount"], axis=1)

# Reorder so target is last
cols = [c for c in df_model.columns if c != "Class"] + ["Class"]
df_model = df_model[cols]

# 3.3 Train–test split (stratified to preserve the fraud ratio)
X = df_model.drop("Class", axis=1)
y = df_model["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)

print("Train shape:", X_train.shape, "| Test shape:", X_test.shape)
print("Train fraud ratio: {:.4f}%".format(y_train.mean() * 100))
print("Test  fraud ratio: {:.4f}%".format(y_test.mean() * 100))

# ============================================================
# 4. CLASS IMBALANCE HANDLING
# ============================================================
# We will compare three strategies inside the modelling loop:
#   (a) No balancing (baseline)
#   (b) SMOTE oversampling
#   (c) NearMiss undersampling
#
# Resampling is applied ONLY on the training set to avoid leakage.
# For cross-validation we use imblearn pipelines.

# ============================================================
# 5. MODEL BUILDING & EVALUATION
# ============================================================

# 5.1 Define base classifiers
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1),
    "SVM (RBF)":           SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE),
    "Decision Tree":       DecisionTreeClassifier(random_state=RANDOM_STATE),
    "k-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
}

# 5.2 Helper: evaluate a fitted model on the held-out test set
def evaluate_model(name, model, X_tr, y_tr, X_te, y_te):
    y_pred  = model.predict(X_te)
    y_proba = model.predict_proba(X_te)[:, 1] if hasattr(model, "predict_proba") else None

    acc  = accuracy_score(y_te, y_pred)
    prec = precision_score(y_te, y_pred, zero_division=0)
    rec  = recall_score(y_te, y_pred, zero_division=0)
    f1   = f1_score(y_te, y_pred, zero_division=0)
    auc  = roc_auc_score(y_te, y_proba) if y_proba is not None else np.nan

    print(f"\n{'='*55}")
    print(f"  {name}")
    print(f"{'='*55}")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall   : {rec:.4f}")
    print(f"  F1-score : {f1:.4f}")
    print(f"  ROC-AUC  : {auc:.4f}")
    print(classification_report(y_te, y_pred, target_names=["Legit", "Fraud"], zero_division=0))

    # ROC curve
    if y_proba is not None:
        fpr, tpr, _ = roc_curve(y_te, y_proba)
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
        plt.plot([0, 1], [0, 1], "k--", alpha=0.6)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve – {name}")
        plt.legend()
        plt.tight_layout()
        plt.show()

    return {"model": name, "accuracy": acc, "precision": prec,
            "recall": rec, "f1": f1, "roc_auc": auc}

# 5.3 Strategy comparison: Baseline vs SMOTE vs NearMiss
results = []

strategies = {
    "Baseline (no balancing)": None,
    "SMOTE (oversampling)":    SMOTE(random_state=RANDOM_STATE),
    "NearMiss (undersampling)": NearMiss(version=1, n_neighbors=3),
}

for strategy_name, sampler in strategies.items():
    print(f"\n\n{'#'*70}")
    print(f"  STRATEGY: {strategy_name}")
    print(f"{'#'*70}")

    # Resample training data (if a sampler is provided)
    if sampler is not None:
        X_res, y_res = sampler.fit_resample(X_train, y_train)
        print("After resampling – class counts:", Counter(y_res))
    else:
        X_res, y_res = X_train, y_train

    for model_name, model in models.items():
        model.fit(X_res, y_res)
        res = evaluate_model(f"{model_name} | {strategy_name}",
                             model, X_res, y_res, X_test, y_test)
        results.append(res)

# ============================================================
# 6. CROSS-VALIDATION (robust performance assessment)
# ============================================================
# We cross-validate the two most promising models using imblearn pipelines.
# This prevents data leakage: SMOTE/NearMiss is fitted inside each fold.

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

cv_models = {
    "Random Forest + SMOTE": ImbPipeline([
        ("smote", SMOTE(random_state=RANDOM_STATE)),
        ("clf",  RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)),
    ]),
    "Logistic Regression + SMOTE": ImbPipeline([
        ("smote", SMOTE(random_state=RANDOM_STATE)),
        ("clf",  LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
    ]),
}

print("\n\n" + "="*70)
print("  5-FOLD CROSS-VALIDATION (ROC-AUC)")
print("="*70)

for name, pipeline in cv_models.items():
    scores = cross_val_score(pipeline, X_train, y_train, cv=cv,
                             scoring="roc_auc", n_jobs=-1)
    print(f"{name:35s}  ROC-AUC = {scores.mean():.4f} ± {scores.std():.4f}")

# ============================================================
# 7. COMPARATIVE SUMMARY
# ============================================================
results_df = pd.DataFrame(results).sort_values("roc_auc", ascending=False)
print("\n\n" + "="*70)
print("  FINAL MODEL COMPARISON (sorted by ROC-AUC)")
print("="*70)
print(results_df.to_string(index=False))

# Visual comparison of key metrics
metrics_to_plot = ["accuracy", "precision", "recall", "f1", "roc_auc"]
fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(20, 5))
for ax, metric in zip(axes, metrics_to_plot):
    top = results_df.nlargest(8, metric)
    ax.barh(top["model"], top[metric], color="teal")
    ax.set_title(metric.replace("_", " ").title())
    ax.set_xlim(0, 1)
plt.tight_layout()
plt.show()

# ============================================================
# 8. MODEL SELECTION & DEPLOYMENT CONSIDERATIONS
# ============================================================
best = results_df.iloc[0]
print("\n\n" + "="*70)
print("  RECOMMENDED MODEL")
print("="*70)
print(f"  Model      : {best['model']}")
print(f"  ROC-AUC    : {best['roc_auc']:.4f}")
print(f"  Recall     : {best['recall']:.4f}")
print(f"  Precision  : {best['precision']:.4f}")
print(f"  F1-score   : {best['f1']:.4f}")
print("\n  Recall is prioritised in fraud detection because missing a")
print("  fraudulent transaction (false negative) is far more costly")
print("  than flagging a legitimate one (false positive).")