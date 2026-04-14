"""
dashboard/app.py
----------------
Streamlit dashboard - 4 pages:
  1. Overview     - live KPIs from PostgreSQL
  2. Predict      - single transaction + CSV batch
  3. Performance  - PR curve, ROC, confusion matrix
  4. Experiments  - MLflow run comparison
"""

import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# Config
API_URL    = os.getenv("API_URL",             "http://localhost:8000")
DB_URL     = os.getenv("DATABASE_URL",        "postgresql://fraud_user:fraud_pass@localhost:5432/fraud_db")
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")

st.set_page_config(
    page_title            = "Fraud Detection Dashboard",
    page_icon             = "shield",
    layout                = "wide",
    initial_sidebar_state = "expanded",
)

st.markdown("""
<style>
    .fraud-tag {
        background: #e94560;
        color: white;
        padding: 2px 10px;
        border-radius: 20px;
        font-weight: 600;
    }
    .legit-tag {
        background: #0f9b58;
        color: white;
        padding: 2px 10px;
        border-radius: 20px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════

@st.cache_data(ttl=30)
def fetch_db_stats():
    try:
        from sqlalchemy import create_engine, text
        engine = create_engine(DB_URL, pool_pre_ping=True)
        with engine.connect() as conn:
            total    = conn.execute(text("SELECT COUNT(*) FROM predictions")).scalar()
            fraud    = conn.execute(text("SELECT COUNT(*) FROM predictions WHERE prediction='FRAUD'")).scalar()
            avg_prob = conn.execute(text("SELECT AVG(fraud_probability) FROM predictions")).scalar()
            recent   = pd.read_sql(
                "SELECT created_at, prediction, fraud_probability, amount, confidence "
                "FROM predictions ORDER BY created_at DESC LIMIT 50",
                conn,
            )
        return {
            "total":    total    or 0,
            "fraud":    fraud    or 0,
            "legit":    (total   or 0) - (fraud or 0),
            "avg_prob": float(avg_prob) if avg_prob else 0.0,
            "recent":   recent,
        }
    except Exception as e:
        st.warning("DB not connected: " + str(e))
        return {
            "total": 0, "fraud": 0, "legit": 0,
            "avg_prob": 0.0, "recent": pd.DataFrame(),
        }


@st.cache_data(ttl=60)
def fetch_mlflow_runs():
    try:
        import mlflow
        mlflow.set_tracking_uri(MLFLOW_URI)
        client = mlflow.MlflowClient()
        exp    = client.get_experiment_by_name("fraud-detection")
        if exp is None:
            return pd.DataFrame()
        runs = client.search_runs(
            experiment_ids=[exp.experiment_id],
            order_by=["metrics.pr_auc DESC"],
        )
        records = []
        for r in runs:
            records.append({
                "Run Name":  r.data.tags.get("mlflow.runName", r.info.run_id[:8]),
                "PR-AUC":    r.data.metrics.get("pr_auc",    0),
                "ROC-AUC":   r.data.metrics.get("roc_auc",   0),
                "F1":        r.data.metrics.get("f1",        0),
                "Precision": r.data.metrics.get("precision", 0),
                "Recall":    r.data.metrics.get("recall",    0),
                "Threshold": r.data.metrics.get("threshold", 0.5),
                "Status":    r.info.status,
            })
        return pd.DataFrame(records)
    except Exception as e:
        st.warning("MLflow not connected: " + str(e))
        return pd.DataFrame()


def call_health():
    try:
        r = requests.get(API_URL + "/health", timeout=5)
        return r.json()
    except Exception:
        return {
            "status":       "unreachable",
            "model_loaded": False,
            "db_connected": False,
            "api_version":  "-",
        }


def call_predict(payload):
    try:
        r = requests.post(API_URL + "/predict", json=payload, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error("API error: " + str(e))
        return None


# ════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## Fraud Detection")
    st.markdown("---")

    page = st.radio(
        "Navigate",
        ["Overview", "Predict", "Performance", "Experiments"],
        label_visibility="collapsed",
    )

    st.markdown("---")

    health = call_health()

    api_status = health.get("status", "unknown").upper()
    model_ok   = health.get("model_loaded", False)
    db_ok      = health.get("db_connected", False)
    version    = health.get("api_version", "-")

    if health.get("status") == "healthy":
        st.success("API: " + api_status)
    else:
        st.error("API: " + api_status)

    st.write("Model: " + ("Loaded" if model_ok else "Not loaded"))
    st.write("DB: "    + ("Connected" if db_ok  else "Not connected"))
    st.write("Version: " + version)

    st.markdown("---")
    if st.button("Refresh"):
        st.cache_data.clear()
        st.rerun()


# ════════════════════════════════════════════════════════════
# PAGE 1 - OVERVIEW
# ════════════════════════════════════════════════════════════

if page == "Overview":
    st.title("Overview Dashboard")
    st.caption("Live statistics from PostgreSQL — refreshes every 30s")

    stats      = fetch_db_stats()
    fraud_rate = (stats["fraud"] / stats["total"] * 100) if stats["total"] > 0 else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Predictions", str(stats["total"]))
    c2.metric("Fraud Detected",    str(stats["fraud"]))
    c3.metric("Fraud Rate",        str(round(fraud_rate, 2)) + "%")
    c4.metric("Avg Fraud Score",   str(round(stats["avg_prob"], 3)))

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Prediction Split")
        if stats["total"] > 0:
            fig = go.Figure(data=[go.Pie(
                labels        = ["LEGIT", "FRAUD"],
                values        = [stats["legit"], stats["fraud"]],
                hole          = 0.5,
                marker_colors = ["#0f9b58", "#e94560"],
            )])
            fig.update_layout(height=300, margin=dict(t=20, b=20))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No predictions yet. Use the Predict page.")

    with col2:
        st.subheader("Recent Predictions")
        recent = stats.get("recent", pd.DataFrame())
        if not recent.empty:
            st.dataframe(recent, use_container_width=True, height=300)
        else:
            st.info("No prediction history yet.")

    if not recent.empty and "fraud_probability" in recent.columns:
        st.subheader("Score Distribution")
        fig2 = px.histogram(
            recent,
            x                       = "fraud_probability",
            nbins                   = 30,
            color_discrete_sequence = ["#2E75B6"],
        )
        fig2.add_vline(
            x               = 0.5,
            line_dash       = "dash",
            line_color      = "red",
            annotation_text = "Threshold = 0.5",
        )
        fig2.update_layout(height=280)
        st.plotly_chart(fig2, use_container_width=True)


# ════════════════════════════════════════════════════════════
# PAGE 2 - PREDICT
# ════════════════════════════════════════════════════════════

elif page == "Predict":
    st.title("Predict Fraud")

    tab1, tab2 = st.tabs(["Single Transaction", "Batch CSV"])

    with tab1:
        st.subheader("Enter Transaction Features")
        st.info("V1-V28 are PCA features. Set all to 0.0 for a quick demo.")

        col1, col2, col3 = st.columns(3)
        with col1:
            amount = st.number_input("Amount ($)", min_value=0.0, value=149.62)
            time_s = st.number_input("Time (s)",   min_value=0.0, value=3600.0)
            v1  = st.number_input("V1",  value=-1.3598)
            v2  = st.number_input("V2",  value=-0.0728)
            v3  = st.number_input("V3",  value=2.5363)
            v4  = st.number_input("V4",  value=1.3782)
            v5  = st.number_input("V5",  value=0.0)
            v6  = st.number_input("V6",  value=0.0)
            v7  = st.number_input("V7",  value=0.0)
            v8  = st.number_input("V8",  value=0.0)
        with col2:
            v9  = st.number_input("V9",  value=0.0)
            v10 = st.number_input("V10", value=0.0)
            v11 = st.number_input("V11", value=0.0)
            v12 = st.number_input("V12", value=0.0)
            v13 = st.number_input("V13", value=0.0)
            v14 = st.number_input("V14", value=-0.3111)
            v15 = st.number_input("V15", value=0.0)
            v16 = st.number_input("V16", value=0.0)
            v17 = st.number_input("V17", value=-0.5535)
        with col3:
            v18 = st.number_input("V18", value=0.0)
            v19 = st.number_input("V19", value=0.0)
            v20 = st.number_input("V20", value=0.0)
            v21 = st.number_input("V21", value=0.0)
            v22 = st.number_input("V22", value=0.0)
            v23 = st.number_input("V23", value=0.0)
            v24 = st.number_input("V24", value=0.0)
            v25 = st.number_input("V25", value=0.0)
            v26 = st.number_input("V26", value=0.0)
            v27 = st.number_input("V27", value=0.0)
            v28 = st.number_input("V28", value=0.0)

        if st.button("Predict", type="primary", use_container_width=True):
            payload = {
                "Amount": amount, "Time": time_s,
                "V1": v1,   "V2": v2,   "V3": v3,   "V4": v4,
                "V5": v5,   "V6": v6,   "V7": v7,   "V8": v8,
                "V9": v9,   "V10": v10, "V11": v11, "V12": v12,
                "V13": v13, "V14": v14, "V15": v15, "V16": v16,
                "V17": v17, "V18": v18, "V19": v19, "V20": v20,
                "V21": v21, "V22": v22, "V23": v23, "V24": v24,
                "V25": v25, "V26": v26, "V27": v27, "V28": v28,
            }
            with st.spinner("Scoring..."):
                result = call_predict(payload)

            if result:
                if result["prediction"] == "FRAUD":
                    st.error("FRAUD DETECTED - Score: " + str(result["fraud_probability"]))
                else:
                    st.success("LEGITIMATE - Score: " + str(result["fraud_probability"]))

                a, b, c = st.columns(3)
                a.metric("Fraud Score", str(result["fraud_probability"]))
                b.metric("Confidence",  result["confidence"])
                c.metric("Latency",     str(result["latency_ms"]) + " ms")
                st.caption("Transaction ID: " + result["transaction_id"])

    with tab2:
        st.subheader("Batch Prediction via CSV")
        st.info("Upload a CSV with columns V1-V28, Amount, Time.")

        uploaded = st.file_uploader("Upload CSV", type=["csv"])
        if uploaded:
            df_up = pd.read_csv(uploaded)
            st.dataframe(df_up.head(), use_container_width=True)
            st.caption(str(len(df_up)) + " rows loaded")

            if st.button("Run Batch", type="primary"):
                valid_cols = [("V" + str(i)) for i in range(1, 29)] + ["Amount", "Time"]
                records    = []
                for _, row in df_up.iterrows():
                    rec = {col: float(row[col]) for col in valid_cols if col in df_up.columns}
                    records.append(rec)

                try:
                    with st.spinner("Scoring " + str(len(records)) + " rows..."):
                        resp = requests.post(
                            API_URL + "/predict/batch",
                            json    = {"transactions": records},
                            timeout = 120,
                        )
                        resp.raise_for_status()
                        batch = resp.json()

                    c1, c2, c3 = st.columns(3)
                    c1.metric("Total",      str(batch["total"]))
                    c2.metric("Fraud",      str(batch["fraud_count"]))
                    c3.metric("Fraud Rate", str(round(batch["fraud_rate"] * 100, 2)) + "%")

                    out_df = pd.DataFrame(batch["predictions"])
                    st.dataframe(out_df, use_container_width=True, height=400)

                    st.download_button(
                        "Download Results",
                        data      = out_df.to_csv(index=False),
                        file_name = "fraud_predictions.csv",
                        mime      = "text/csv",
                    )
                except Exception as e:
                    st.error("Batch failed: " + str(e))


# ════════════════════════════════════════════════════════════
# PAGE 3 - PERFORMANCE
# ════════════════════════════════════════════════════════════

elif page == "Performance":
    st.title("Model Performance")
    st.info("Run python -m src.train first to generate model artifacts.")

    try:
        import joblib
        from sklearn.metrics import (
            precision_recall_curve,
            roc_curve,
            confusion_matrix,
            average_precision_score,
            roc_auc_score,
            f1_score,
        )

        model  = joblib.load("models/best_model.pkl")
        X_test = np.load("data/processed/X_test.npy")
        y_test = np.load("data/processed/y_test.npy")
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        pr_auc  = average_precision_score(y_test, y_prob)
        roc_auc = roc_auc_score(y_test, y_prob)

        st.success(
            "Model loaded — PR-AUC: " + str(round(pr_auc, 4)) +
            " | ROC-AUC: " + str(round(roc_auc, 4))
        )

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Precision-Recall Curve")
            prec, rec, _ = precision_recall_curve(y_test, y_prob)
            fig_pr = go.Figure()
            fig_pr.add_trace(go.Scatter(
                x=rec, y=prec, mode="lines",
                name="PR-AUC = " + str(round(pr_auc, 3)),
                line=dict(color="#2E75B6", width=2),
            ))
            fig_pr.add_hline(
                y=y_test.mean(),
                line_dash="dash",
                line_color="red",
                annotation_text="Random baseline",
            )
            fig_pr.update_layout(height=350, xaxis_title="Recall", yaxis_title="Precision")
            st.plotly_chart(fig_pr, use_container_width=True)

        with col2:
            st.subheader("ROC Curve")
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(
                x=fpr, y=tpr, mode="lines",
                name="ROC-AUC = " + str(round(roc_auc, 3)),
                line=dict(color="#e94560", width=2),
            ))
            fig_roc.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1], mode="lines",
                line=dict(dash="dash", color="gray"),
                name="Random",
            ))
            fig_roc.update_layout(height=350, xaxis_title="FPR", yaxis_title="TPR")
            st.plotly_chart(fig_roc, use_container_width=True)

        st.subheader("Confusion Matrix")
        cm     = confusion_matrix(y_test, y_pred)
        fig_cm = px.imshow(
            cm,
            labels                 = dict(x="Predicted", y="Actual"),
            x                      = ["LEGIT", "FRAUD"],
            y                      = ["LEGIT", "FRAUD"],
            color_continuous_scale = "Blues",
            text_auto              = True,
        )
        fig_cm.update_layout(height=350)
        st.plotly_chart(fig_cm, use_container_width=True)

        st.subheader("Threshold vs F1 Score")
        thresholds = np.arange(0.1, 0.9, 0.02)
        f1s        = [
            f1_score(y_test, (y_prob >= t).astype(int), zero_division=0)
            for t in thresholds
        ]
        fig_t = go.Figure()
        fig_t.add_trace(go.Scatter(
            x=thresholds, y=f1s, mode="lines",
            name="F1 Score",
            line=dict(color="#2E75B6", width=2),
        ))
        fig_t.add_vline(
            x=0.5,
            line_dash="dash",
            line_color="red",
            annotation_text="Default (0.5)",
        )
        fig_t.update_layout(height=300, xaxis_title="Threshold", yaxis_title="F1 Score")
        st.plotly_chart(fig_t, use_container_width=True)

    except FileNotFoundError:
        st.warning("No model found. Run python -m src.train first.")
    except Exception as e:
        st.error("Error: " + str(e))


# ════════════════════════════════════════════════════════════
# PAGE 4 - EXPERIMENTS
# ════════════════════════════════════════════════════════════

elif page == "Experiments":
    st.title("MLflow Experiments")
    st.caption("Tracking URI: " + MLFLOW_URI)

    runs_df = fetch_mlflow_runs()

    if runs_df.empty:
        st.warning("No runs found. Run python -m src.train to populate.")
        st.markdown("Open MLflow UI: " + MLFLOW_URI)
    else:
        st.subheader("All Runs")
        st.dataframe(
            runs_df.style.highlight_max(
                subset=["PR-AUC", "ROC-AUC", "F1"],
                color="#d9ead3",
            ),
            use_container_width=True,
        )

        st.subheader("Metric Comparison")
        fig_bar = px.bar(
            runs_df.melt(
                id_vars    = ["Run Name"],
                value_vars = ["PR-AUC", "ROC-AUC", "F1"],
            ),
            x       = "Run Name",
            y       = "value",
            color   = "variable",
            barmode = "group",
            height  = 400,
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        st.markdown("MLflow UI: " + MLFLOW_URI)