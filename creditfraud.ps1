# ============================================================
#  create-fraud-project.ps1
#  Creates the full Credit Card Fraud Detection project.
#  Usage:
#    powershell -ExecutionPolicy Bypass -File .\create-fraud-project.ps1
# ============================================================

param(
    [string]$ProjectName = "credit-card-fraud-detection"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

# ---------- helpers -----------------------------------------
function Write-FileNoBom {
    param([string]$Path, [string]$Content)
    $dir = Split-Path -Parent $Path
    if ($dir -and -not (Test-Path $dir)) { New-Item -ItemType Directory -Path $dir -Force | Out-Null }
    $utf8NoBom = New-Object System.Text.UTF8Encoding $false
    [System.IO.File]::WriteAllText($Path, $Content, $utf8NoBom)
    Write-Host "  ✔ $Path" -ForegroundColor Green
}

function Write-Touch {
    param([string]$Path)
    $dir = Split-Path -Parent $Path
    if ($dir -and -not (Test-Path $dir)) { New-Item -ItemType Directory -Path $dir -Force | Out-Null }
    if (-not (Test-Path $Path)) { New-Item -ItemType File -Path $Path -Force | Out-Null }
    Write-Host "  ✔ $Path" -ForegroundColor Green
}

# ---------- scaffold ----------------------------------------
$root = Join-Path (Get-Location) $ProjectName
if (Test-Path $root) {
    Write-Host "Folder '$ProjectName' already exists — overwriting files." -ForegroundColor Yellow
} else {
    New-Item -ItemType Directory -Path $root -Force | Out-Null
}
Write-Host "`nCreating project in: $root`n" -ForegroundColor Cyan

Set-Location $root
Write-Host "Creating folders…" -ForegroundColor Cyan
foreach ($d in @("data","outputs","docs")) {
    if (-not (Test-Path $d)) { New-Item -ItemType Directory -Path $d -Force | Out-Null }
}

# ============================================================
#  1. requirements.txt
# ============================================================
$requirements = @'
# ---- Core data ----
numpy>=1.24,<2.0
pandas>=2.0,<3.0

# ---- Visualisation ----
matplotlib>=3.7,<4.0
seaborn>=0.13,<1.0

# ---- Machine learning ----
scikit-learn>=1.3,<2.0
imbalanced-learn>=0.11,<1.0

# ---- Web app ----
streamlit>=1.32,<2.0
'@
Write-Host "`nWriting requirements…" -ForegroundColor Cyan
Write-FileNoBom "requirements.txt" $requirements

# ============================================================
#  2. requirements-dev.txt
# ============================================================
$requirementsDev = @'
-r requirements.txt
black>=24.0
flake8>=7.0
pytest>=8.0
jupyter>=1.0
'@
Write-FileNoBom "requirements-dev.txt" $requirementsDev

# ============================================================
#  3. .gitignore
# ============================================================
$gitignore = @'
# ---- Data (never commit the 150 MB CSV) ----
data/*.csv
*.csv
!sample_data/*.csv

# ---- Outputs ----
outputs/
*.png
*.pdf
*.joblib
*.pkl

# ---- Python ----
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
dist/
*.egg-info/
.eggs/

# ---- Virtual environments ----
.venv/
venv/
env/
ENV/

# ---- Jupyter ----
.ipynb_checkpoints/

# ---- Streamlit ----
.streamlit/secrets.toml

# ---- IDE / OS ----
.vscode/
.idea/
.DS_Store
Thumbs.db

# ---- Testing / coverage ----
.pytest_cache/
.coverage
htmlcov/
'@
Write-Host "`nWriting .gitignore…" -ForegroundColor Cyan
Write-FileNoBom ".gitignore" $gitignore

# ============================================================
#  4. README.md
# ============================================================
$readme = @'
# 💳 Credit Card Fraud Detection

An end-to-end data science project that detects fraudulent credit card transactions
using machine learning, with a companion **Streamlit web app** for interactive
exploration, training, evaluation, and live prediction.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.3x-red)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📌 Overview

Financial fraud is rare but costly — the well-known
[ULB / Kaggle dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
contains **284,807** transactions of which only **492 (≈0.172 %)** are fraudulent.
This extreme class imbalance makes **accuracy a misleading metric** — a model that
always predicts "legitimate" is already 99.83 % accurate but catches zero frauds.

This project therefore focuses on **Recall**, **Precision**, **F1**, and
**ROC-AUC**, and systematically compares models trained under different
imbalance-handling strategies (**SMOTE**, **NearMiss**, no balancing).

---

## ✨ Features

- 📊 **Exploratory Data Analysis** — class distribution, amount histograms,
  correlation heatmap, per-class boxplots, IQR outlier detection.
- ⚙️ **Robust preprocessing** — de-duplication, standard scaling of
  `Time` / `Amount`, leakage-free train/test split (stratified).
- ⚖️ **Imbalance handling** — plug-and-play **SMOTE**, **NearMiss**, or baseline.
- 🤖 **Multiple models** — Logistic Regression, Random Forest, Decision Tree,
  k-Nearest Neighbors.
- 📈 **Evaluation** — Accuracy, Precision, Recall, F1, ROC-AUC, confusion matrix,
  overlaid ROC curves, feature importances.
- 🔁 **5-fold cross-validation** on the best model (SMOTE fitted inside folds — no leakage).
- 🔮 **Live prediction** — score an uploaded CSV or sampled test rows with an
  adjustable fraud-decision threshold.
- 🧪 **Synthetic-data mode** — the app ships with a realistic synthetic dataset so
  you can explore without downloading the 150 MB original.

---

## 🗂️ Repository Structure
