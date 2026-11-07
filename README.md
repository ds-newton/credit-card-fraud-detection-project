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

```
credit-card-fraud-detection/
├── fraud_app.py            # Interactive Streamlit application
├── fraud_detection.py      # Standalone CLI / notebook-style solution
├── requirements.txt        # Python dependencies
├── README.md               # You are here
├── .gitignore              # Ignores data, caches, venvs
├── data/
│   └── .gitkeep            # (place creditcard.csv here)
└── outputs/                # (plots / metric CSVs, auto-created)
```

> **Note:** `creditcard.csv` is intentionally **not** committed (150 MB). See
> [Dataset](#-dataset) below.

---

## 📦 Dataset

The original dataset was released by the **Machine Learning Group at Université
Libre de Bruxelles (ULB)** in collaboration with **Worldline**, and is publicly
hosted on Kaggle:

🔗 **https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud**

| Attribute | Description |
|---|---|
| `Time` | Seconds elapsed since the first transaction |
| `Amount` | Transaction amount |
| `V1–V28` | PCA-transformed anonymised features |
| `Class` | `0` = legitimate, `1` = fraud |

**Quick download via Kaggle CLI:**

```bash
pip install kaggle
kaggle datasets download -d mlg-ulb/creditcardfraud
unzip creditcardfraud.zip -d data/
```

Then run the app and pick **Upload CSV** in the sidebar.

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/credit-card-fraud-detection.git
cd credit-card-fraud-detection
```

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate         # macOS / Linux
.venv\Scripts\activate            # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Launch the Streamlit app

```bash
streamlit run fraud_app.py
```

The app opens at **http://localhost:8501**.

In the sidebar, either:
- **Generate synthetic demo** → works instantly, no download needed, or
- **Upload CSV** → use the real `creditcard.csv`.

### 5. (Optional) Run the standalone script

```bash
python fraud_detection.py
```

Ensure `creditcard.csv` is in the working directory first.

---

## 🖥️ App Walkthrough

| Page | Purpose |
|---|---|
| **🏠 Overview** | Project context + dataset KPIs |
| **📊 Data & EDA** | Preview, describe, plots, outlier summary |
| **🧠 Train Models** | Pick models × imbalance strategies, train, optional 5-fold CV |
| **📈 Evaluate & Compare** | Metrics table, bar charts, ROC curves, confusion matrix, feature importances |
| **🔮 Make Predictions** | Score a new CSV **or** sample test rows, tune threshold, download results |

### Key design principles

- **No data leakage** — scaler + SMOTE are fitted on the training split only,
  and SMOTE runs *inside* each CV fold.
- **Recall-first mindset** — the prediction page exposes the decision threshold
  so you can trade precision for recall depending on the business cost of a
  missed fraud vs. a false alarm.
- **Session state** — trained models, scaler, and results persist across page
  navigation (you only train once).

---

## 🧪 Example Results

On a synthetic 20k-row demo (5 % fraud) with default settings:

| Model | Strategy | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|---|
| Random Forest | SMOTE | 0.998 | 0.98 | 0.96 | 0.97 | 0.999 |
| Logistic Regression | SMOTE | 0.995 | 0.94 | 0.93 | 0.93 | 0.997 |
| Random Forest | Baseline | 0.999 | 1.00 | 0.85 | 0.92 | 0.999 |
| k-NN | NearMiss | 0.980 | 0.70 | 0.88 | 0.78 | 0.985 |

*(Numbers are illustrative — rerun on the full real dataset for authoritative results.)*

Takeaway: the **baseline Random Forest has higher precision but lower recall** —
SMOTE trades a small precision loss for a substantial recall gain, which is
usually the correct trade-off in fraud detection.

---

## 🛠️ Tech Stack

- **Python 3.10+**
- **Data:** `pandas`, `numpy`
- **Visualisation:** `matplotlib`, `seaborn`
- **ML:** `scikit-learn`, `imbalanced-learn`
- **App:** `streamlit`

---

## 🤝 Contributing

Contributions are welcome!

1. Fork the repo
2. Create a branch: `git checkout -b feature/my-feature`
3. Commit: `git commit -m "Add my feature"`
4. Push: `git push origin feature/my-feature`
5. Open a Pull Request

Please run `black .` and `flake8` before submitting (optional but appreciated).

---

## 📜 License

Released under the **MIT License**. See [`LICENSE`](LICENSE) for details.

The **dataset** is governed by the original authors' terms — see the
[Kaggle page](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) for details.

---

## 🙏 Acknowledgements

- **Andrea Dal Pozzolo, Olivier Caelen, Reid A. Johnson, Gianluca Bontempi** —
  original dataset authors (*"Calibrating Probability with Undersampling for
  Unbalanced Classification"*, IEEE SSCI 2015).
- **Machine Learning Group, Université Libre de Bruxelles (ULB)** and **Worldline**.
- [Kaggle](https://www.kaggle.com/) for hosting the dataset.

---

## 📬 Contact

- **Author:** Newton Mwalongo
- **Email:** newtoncyta@gmail.com
- **LinkedIn:** [linkedin.com/in/blessednewton](https://linkedin.com/in/blessednewton)

⭐ If you find this useful, please **star the repo**!