"""
Credit Card Fraud Detection — Streamlit App
Run:  streamlit run fraud_app.py
"""

import warnings
warnings.filterwarnings("ignore")

import io
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report,
)
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss

# ---------------------------------------------------------------------
# Page config & styling
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

sns.set_style("whitegrid")
RANDOM_STATE = 42

# ---------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------
for key, default in {
    "df": None,
    "trained_models": {},        # name -> {"model": ..., "metrics": ...}
    "results_df": None,
    "scaler": None,
    "feature_cols": None,
    "X_test": None,
    "y_test": None,
    "y_test_proba": {},          # name -> proba array
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def generate_synthetic_data(n_samples: int = 20_000, fraud_rate: float = 0.005, seed: int = 42):
    """Generate a credit-card-like synthetic dataset for demo purposes."""
    rng = np.random.default_rng(seed)
    n_fraud = max(int(n_samples * fraud_rate), 20)
    n_legit = n_samples - n_fraud

    V_legit = rng.normal(0, 0.9, size=(n_legit, 28))
    V_fraud = rng.normal(0, 1.4, size=(n_fraud, 28))
    for i in [0, 2, 3, 6, 9, 11, 13, 16]:
        V_fraud[:, i] += rng.choice([-1.8, 1.8], size=n_fraud)

    V = np.vstack([V_legit, V_fraud])
    Amount = np.concatenate([
        rng.exponential(scale=80, size=n_legit),
        rng.exponential(scale=140, size=n_fraud),
    ])
    Time = np.sort(rng.uniform(0, 172800, size=n_samples))
    Class = np.concatenate([np.zeros(n_legit, dtype=int), np.ones(n_fraud, dtype=int)])

    df = pd.DataFrame(V, columns=[f"V{i}" for i in range(1, 29)])
    df["Time"] = Time
    df["Amount"] = Amount
    df["Class"] = Class
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_csv_bytes(data_bytes: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(data_bytes))


def preprocess(df: pd.DataFrame):
    """Drop duplicates, scale Time/Amount, return X, y, scaler, feature_cols."""
    df = df.drop_duplicates().reset_index(drop=True)
    scaler = StandardScaler()
    df = df.copy()
    df["scaled_amount"] = scaler.fit_transform(df["Amount"].values.reshape(-1, 1))
    df["scaled_time"]   = scaler.fit_transform(df["Time"].values.reshape(-1, 1))
    df = df.drop(["Time", "Amount"], axis=1)
    cols = [c for c in df.columns if c != "Class"] + ["Class"]
    df = df[cols]
    X = df.drop("Class", axis=1)
    y = df["Class"]
    return X, y, scaler, list(X.columns)


MODEL_FACTORY = {
    "Logistic Regression": lambda: LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    "Random Forest":       lambda: RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1),
    "Decision Tree":       lambda: DecisionTreeClassifier(random_state=RANDOM_STATE),
    "k-Nearest Neighbors": lambda: KNeighborsClassifier(n_neighbors=5),
}

SAMPLER_FACTORY = {
    "Baseline (none)": lambda: None,
    "SMOTE":           lambda: SMOTE(random_state=RANDOM_STATE),
    "NearMiss":        lambda: NearMiss(version=1, n_neighbors=3),
}


def apply_sampler(X_tr, y_tr, sampler_name):
    sampler = SAMPLER_FACTORY[sampler_name]()
    if sampler is None:
        return X_tr, y_tr
    return sampler.fit_resample(X_tr, y_tr)


def train_one(model_name, sampler_name, X_train, y_train, X_test, y_test):
    """Train a single model with the chosen sampler and return metrics + predictions."""
    model = MODEL_FACTORY[model_name]()
    X_res, y_res = apply_sampler(X_train, y_train, sampler_name)

    model.fit(X_res, y_res)
    y_pred = model.predict(X_test)
    y_proba = (
        model.predict_proba(X_test)[:, 1]
        if hasattr(model, "predict_proba") else None
    )

    metrics = {
        "Model":     model_name,
        "Strategy":  sampler_name,
        "Accuracy":  accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall":    recall_score(y_test, y_pred, zero_division=0),
        "F1":        f1_score(y_test, y_pred, zero_division=0),
        "ROC-AUC":   roc_auc_score(y_test, y_proba) if y_proba is not None else np.nan,
    }
    return model, metrics, y_pred, y_proba


# ---------------------------------------------------------------------
# Sidebar: data loading
# ---------------------------------------------------------------------
st.sidebar.title("💳 Fraud Detection")
st.sidebar.markdown("---")

data_source = st.sidebar.radio(
    "Data source",
    ["Upload CSV", "Generate synthetic demo", "Load sample from URL"],
    index=1,
)

if data_source == "Upload CSV":
    uploaded = st.sidebar.file_uploader(
        "Upload creditcard.csv", type=["csv"],
        help="Expected columns: Time, Amount, V1–V28, Class",
    )
    if uploaded is not None:
        with st.spinner("Loading CSV…"):
            st.session_state.df = load_csv_bytes(uploaded.getvalue())
        st.sidebar.success(f"Loaded {len(st.session_state.df):,} rows")

elif data_source == "Generate synthetic demo":
    n = st.sidebar.slider("Number of rows", 5_000, 100_000, 20_000, 5_000)
    if st.sidebar.button("Generate dataset", type="primary"):
        with st.spinner("Generating synthetic data…"):
            st.session_state.df = generate_synthetic_data(n_samples=n)
        st.sidebar.success(f"Generated {len(st.session_state.df):,} rows")

elif data_source == "Load sample from URL":
    st.sidebar.info(
        "Place `creditcard.csv` (from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)) "
        "in the app directory and use **Upload CSV** instead — the real file is ~150 MB."
    )

st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigate",
    ["🏠 Overview", "📊 Data & EDA", "🧠 Train Models",
     "📈 Evaluate & Compare", "🔮 Make Predictions"],
)

# ---------------------------------------------------------------------
# PAGE: Overview
# ---------------------------------------------------------------------
if page == "🏠 Overview":
    st.title("Credit Card Fraud Detection")
    st.markdown(
        """
        ### Business context
        Credit-card fraud costs the industry billions every year. Because fraud is
        **extremely rare (< 1 % of transactions)**, standard accuracy is misleading —
        a model that predicts "legitimate" every time is already 99.8 % accurate.

        ### What this app does
        | Section | Purpose |
        |---|---|
        | **Data & EDA** | Inspect dataset structure, class imbalance, correlations, outliers |
        | **Train Models** | Train LR / RF / DT / kNN under 3 class-imbalance strategies |
        | **Evaluate & Compare** | ROC curves, confusion matrices, metric comparison |
        | **Make Predictions** | Score new transactions with a chosen trained model |

        ### Dataset
        The ULB / Kaggle **creditcard.csv** dataset contains 284,807 European
        card transactions from September 2013, with 492 frauds (~0.172 %).
        Features `V1–V28` are PCA-transformed; `Time` and `Amount` are raw.

        Use the sidebar to load data, then work through the pages in order.
        """
    )
    if st.session_state.df is not None:
        df = st.session_state.df
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rows", f"{len(df):,}")
        c2.metric("Features", f"{df.shape[1]}")
        c3.metric("Frauds", f"{int(df['Class'].sum()):,}")
        c4.metric("Fraud rate", f"{df['Class'].mean()*100:.3f}%")

# ---------------------------------------------------------------------
# PAGE: Data & EDA
# ---------------------------------------------------------------------
elif page == "📊 Data & EDA":
    st.title("Data & Exploratory Analysis")

    if st.session_state.df is None:
        st.warning("⬅️ Load a dataset from the sidebar first.")
        st.stop()

    df = st.session_state.df
    st.subheader("Preview")
    st.dataframe(df.head(20), use_container_width=True)

    st.subheader("Summary")
    st.write(df.describe().T)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Class distribution")
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.countplot(x="Class", data=df, palette="Set2", ax=ax)
        ax.set_yscale("log")
        ax.set_title("0 = Legit, 1 = Fraud (log scale)")
        st.pyplot(fig)
        plt.close(fig)

    with col2:
        st.subheader("Amount distribution")
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.histplot(df[df["Class"] == 0]["Amount"], bins=60, color="steelblue",
                     label="Legit", ax=ax)
        sns.histplot(df[df["Class"] == 1]["Amount"], bins=60, color="crimson",
                     label="Fraud", ax=ax)
        ax.set_xlim(0, df["Amount"].quantile(0.99))
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)

    with st.expander("Correlation heatmap (V1–V28 + Amount + Class)"):
        fig, ax = plt.subplots(figsize=(14, 10))
        corr = df.corr(numeric_only=True)
        sns.heatmap(corr, cmap="coolwarm", center=0, ax=ax,
                    cbar_kws={"shrink": 0.7})
        st.pyplot(fig)
        plt.close(fig)

    with st.expander("Boxplots — outliers by class"):
        features = ["Amount", "V1", "V3", "V7", "V10", "V12", "V14", "V17"]
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        for ax, feat in zip(axes.ravel(), features):
            sns.boxplot(x="Class", y=feat, data=df, ax=ax, palette="Set3")
            ax.set_title(feat)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    st.subheader("IQR outlier summary (Amount)")
    Q1, Q3 = df["Amount"].quantile([0.25, 0.75])
    IQR = Q3 - Q1
    lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    n_out = ((df["Amount"] < lower) | (df["Amount"] > upper)).sum()
    st.write(f"Outliers in `Amount` (1.5 × IQR): **{n_out:,}** "
             f"({n_out / len(df) * 100:.2f} % of rows)")

# ---------------------------------------------------------------------
# PAGE: Train Models
# ---------------------------------------------------------------------
elif page == "🧠 Train Models":
    st.title("Train Models")

    if st.session_state.df is None:
        st.warning("⬅️ Load a dataset from the sidebar first.")
        st.stop()

    df = st.session_state.df

    col1, col2, col3 = st.columns(3)
    with col1:
        selected_models = st.multiselect(
            "Models",
            list(MODEL_FACTORY.keys()),
            default=["Logistic Regression", "Random Forest"],
        )
    with col2:
        selected_strategies = st.multiselect(
            "Imbalance strategies",
            list(SAMPLER_FACTORY.keys()),
            default=["Baseline (none)", "SMOTE"],
        )
    with col3:
        test_size = st.slider("Test split (%)", 10, 40, 20, 5)

    sample_frac = st.slider(
        "Use fraction of data for training (faster iteration on the full 284k-row file)",
        0.05, 1.0, 1.0, 0.05,
    )

    run_cv = st.checkbox("Also run 5-fold cross-validation on the best model", value=False)

    if st.button("🚀 Train", type="primary",
                 disabled=not selected_models or not selected_strategies):
        with st.spinner("Preprocessing data…"):
            sub = df.sample(frac=sample_frac, random_state=RANDOM_STATE) \
                    if sample_frac < 1.0 else df
            X, y, scaler, feature_cols = preprocess(sub)

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size / 100,
                stratify=y, random_state=RANDOM_STATE,
            )
            st.session_state.scaler = scaler
            st.session_state.feature_cols = feature_cols
            st.session_state.X_test = X_test
            st.session_state.y_test = y_test
            st.session_state.trained_models = {}
            st.session_state.y_test_proba = {}
            results = []

        total = len(selected_models) * len(selected_strategies)
        bar = st.progress(0.0, text="Training…")
        done = 0

        for model_name in selected_models:
            for strat in selected_strategies:
                key = f"{model_name} | {strat}"
                with st.spinner(f"Training **{key}**…"):
                    model, metrics, _, proba = train_one(
                        model_name, strat, X_train, y_train, X_test, y_test
                    )
                st.session_state.trained_models[key] = model
                if proba is not None:
                    st.session_state.y_test_proba[key] = proba
                results.append(metrics)
                done += 1
                bar.progress(done / total,
                             text=f"Trained {done}/{total}: {key}")

        st.session_state.results_df = (
            pd.DataFrame(results)
              .sort_values("ROC-AUC", ascending=False)
              .reset_index(drop=True)
        )

        if run_cv and len(selected_models) > 0:
            from imblearn.pipeline import Pipeline as ImbPipeline
            best_row = st.session_state.results_df.iloc[0]
            best_key = f"{best_row['Model']} | {best_row['Strategy']}"
            st.info(f"Running 5-fold CV on **{best_key}**…")

            pipeline_steps = []
            if best_row["Strategy"] != "Baseline (none)":
                pipeline_steps.append(("sampler", SAMPLER_FACTORY[best_row["Strategy"]]()))
            pipeline_steps.append(("clf", MODEL_FACTORY[best_row["Model"]]()))

            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
            try:
                scores = cross_val_score(
                    ImbPipeline(pipeline_steps), X, y,
                    cv=cv, scoring="roc_auc", n_jobs=-1,
                )
                st.success(
                    f"5-fold ROC-AUC — mean **{scores.mean():.4f}**, "
                    f"std **{scores.std():.4f}**"
                )
            except Exception as e:
                st.warning(f"Cross-validation failed: {e}")

        st.success(f"Trained {len(results)} model(s). Go to **Evaluate & Compare**.")

# ---------------------------------------------------------------------
# PAGE: Evaluate & Compare
# ---------------------------------------------------------------------
elif page == "📈 Evaluate & Compare":
    st.title("Evaluate & Compare")

    results_df = st.session_state.results_df
    if results_df is None or results_df.empty:
        st.warning("Train at least one model first (**Train Models**).")
        st.stop()

    st.subheader("Model comparison")
    st.dataframe(
        results_df.style.format({
            "Accuracy": "{:.4f}", "Precision": "{:.4f}",
            "Recall": "{:.4f}", "F1": "{:.4f}", "ROC-AUC": "{:.4f}",
        }),
        use_container_width=True,
    )

    # Best model highlight
    best = results_df.iloc[0]
    st.markdown(
        f"**Recommended model:** `{best['Model']} | {best['Strategy']}`  "
        f"— ROC-AUC **{best['ROC-AUC']:.4f}**, Recall **{best['Recall']:.4f}**"
    )

    # Metric comparison bar chart
    st.subheader("Metric comparison")
    metrics = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(20, 5))
    labels = results_df["Model"] + " | " + results_df["Strategy"]
    for ax, m in zip(axes, metrics):
        ax.barh(labels, results_df[m], color="teal")
        ax.set_title(m)
        ax.set_xlim(0, 1)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # ROC curves overlay
    probas = st.session_state.y_test_proba
    if probas:
        st.subheader("ROC curves")
        fig, ax = plt.subplots(figsize=(7, 6))
        for key, proba in probas.items():
            fpr, tpr, _ = roc_curve(st.session_state.y_test, proba)
            auc = roc_auc_score(st.session_state.y_test, proba)
            ax.plot(fpr, tpr, label=f"{key} (AUC={auc:.4f})")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.6)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC curves — all trained models")
        ax.legend(fontsize=8, loc="lower right")
        st.pyplot(fig)
        plt.close(fig)

    # Confusion matrix for a selected model
    st.subheader("Confusion matrix")
    chosen = st.selectbox("Choose model", list(st.session_state.trained_models.keys()))
    if chosen:
        model = st.session_state.trained_models[chosen]
        X_test, y_test = st.session_state.X_test, st.session_state.y_test
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)

        c1, c2 = st.columns([1, 1])
        with c1:
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                        xticklabels=["Legit", "Fraud"],
                        yticklabels=["Legit", "Fraud"])
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            ax.set_title(chosen)
            st.pyplot(fig)
            plt.close(fig)
        with c2:
            st.text(classification_report(
                y_test, y_pred, target_names=["Legit", "Fraud"],
                zero_division=0,
            ))

    # Feature importance for tree-based models
    st.subheader("Feature importances (tree-based models only)")
    tree_models = [
        k for k, m in st.session_state.trained_models.items()
        if hasattr(m, "feature_importances_")
    ]
    if tree_models:
        pick = st.selectbox("Select tree model", tree_models)
        model = st.session_state.trained_models[pick]
        imp = pd.Series(model.feature_importances_,
                        index=st.session_state.feature_cols
                        ).sort_values(ascending=False).head(15)
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(x=imp.values, y=imp.index, ax=ax, palette="viridis")
        ax.set_title(f"Top 15 features — {pick}")
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.info("Train a Random Forest or Decision Tree to see feature importances.")

# ---------------------------------------------------------------------
# PAGE: Make Predictions
# ---------------------------------------------------------------------
elif page == "🔮 Make Predictions":
    st.title("Make Predictions")

    if not st.session_state.trained_models:
        st.warning("Train at least one model first (**Train Models**).")
        st.stop()

    model_key = st.selectbox(
        "Model", list(st.session_state.trained_models.keys())
    )
    model = st.session_state.trained_models[model_key]
    scaler = st.session_state.scaler
    feature_cols = st.session_state.feature_cols
    threshold = st.slider(
        "Decision threshold (probability ≥ threshold → fraud)",
        0.0, 1.0, 0.5, 0.01,
        help="Lower threshold → higher recall (catch more fraud) but more false positives.",
    )

    st.markdown("---")
    mode = st.radio(
        "Prediction source",
        ["Upload new transactions CSV", "Sample rows from test set"],
        horizontal=True,
    )

    def score(df_new: pd.DataFrame) -> pd.DataFrame:
        """Scale, reorder columns, and predict."""
        d = df_new.copy()
        # Add scaled columns if raw Time/Amount present
        if "Time" in d.columns and "Amount" in d.columns:
            d["scaled_amount"] = scaler.transform(d["Amount"].values.reshape(-1, 1))
            d["scaled_time"]   = scaler.transform(d["Time"].values.reshape(-1, 1))
            d = d.drop(["Time", "Amount"], axis=1)
        # Ensure feature order matches training
        missing = [c for c in feature_cols if c not in d.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")
        d = d[feature_cols]
        proba = model.predict_proba(d)[:, 1] \
            if hasattr(model, "predict_proba") else None
        if proba is None:
            pred = model.predict(d)
            proba = np.where(pred == 1, 1.0, 0.0)
        else:
            pred = (proba >= threshold).astype(int)
        out = df_new.copy()
        out["fraud_probability"] = proba
        out["prediction"] = np.where(pred == 1, "FRAUD", "LEGIT")
        return out

    if mode == "Upload new transactions CSV":
        up = st.file_uploader(
            "Upload transactions CSV (Time, Amount, V1–V28; Class optional)",
            type=["csv"],
        )
        if up is not None:
            new_df = pd.read_csv(up)
            st.write("Preview:")
            st.dataframe(new_df.head(10), use_container_width=True)
            try:
                with st.spinner("Scoring…"):
                    scored = score(new_df)
                n_fraud = (scored["prediction"] == "FRAUD").sum()
                st.success(
                    f"Scored **{len(scored):,}** rows — "
                    f"**{n_fraud:,}** flagged as fraud "
                    f"({n_fraud / len(scored) * 100:.2f} %)"
                )
                st.dataframe(scored, use_container_width=True)
                st.download_button(
                    "⬇️ Download predictions CSV",
                    scored.to_csv(index=False).encode(),
                    file_name="fraud_predictions.csv",
                    mime="text/csv",
                )
            except Exception as e:
                st.error(f"Prediction failed: {e}")

    else:  # Sample rows from held-out test set
        X_test = st.session_state.X_test
        y_test = st.session_state.y_test
        n = st.slider("How many test rows to score?", 10, min(500, len(X_test)), 50)
        idx = np.random.RandomState(RANDOM_STATE).choice(len(X_test), n, replace=False)
        sample = X_test.iloc[idx].copy()
        # Re-introduce raw Time/Amount for display (undo scaled values roughly)
        display = sample.copy()
        truth = y_test.iloc[idx].values

        proba = model.predict_proba(sample)[:, 1] \
            if hasattr(model, "predict_proba") else None
        pred = (proba >= threshold).astype(int) if proba is not None else model.predict(sample)

        out = display.copy()
        out["actual"] = np.where(truth == 1, "FRAUD", "LEGIT")
        if proba is not None:
            out["fraud_probability"] = np.round(proba, 4)
        out["prediction"] = np.where(pred == 1, "FRAUD", "LEGIT")

        st.dataframe(out[["actual", "fraud_probability", "prediction"]]
                     .head(50), use_container_width=True)

        tp = ((pred == 1) & (truth == 1)).sum()
        fp = ((pred == 1) & (truth == 0)).sum()
        fn = ((pred == 0) & (truth == 1)).sum()
        tn = ((pred == 0) & (truth == 0)).sum()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("True Positives",  tp)
        c2.metric("False Positives", fp)
        c3.metric("False Negatives", fn)
        c4.metric("True Negatives",  tn)