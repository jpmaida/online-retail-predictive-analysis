
import sys
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go

from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    roc_auc_score
)

# ------------------------------------------------------------
# PROJECT PATHS
# ------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

sys.path.insert(0, str(BASE_DIR / "src"))

from high_value_customer import (
    load_dataset,
    build_customer_features,
    REFERENCE_DATE
)

MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

MODEL_CONFIG = {
    "Logistic Regression": {
        "file": MODELS_DIR / "LogisticRegression.joblib",
        "key": "LogisticRegression",
        "color": "#38BDF8"
    },
    "Random Forest": {
        "file": MODELS_DIR / "RandomForest.joblib",
        "key": "RandomForest",
        "color": "#A78BFA"
    }
}

NUMERIC_FEATURES = [
    "Recency",
    "Frequency",
    "Monetary",
    "AvgOrderValue",
    "TotalQuantity",
    "UniqueProducts",
    "AvgDaysBetweenOrders",
    "CustomerTenure",
    "AvgItemsPerOrder",
    "SpendLast30Days",
    "SpendPrevious60Days"
]

# ------------------------------------------------------------
# PAGE CONFIGURATION
# ------------------------------------------------------------

st.set_page_config(
    page_title="HighValue AI",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------------------------------------------
# CUSTOM CSS
# ------------------------------------------------------------

st.markdown("""
<style>

.stApp {
    background-color: #0B1120;
}

[data-testid="stSidebar"] {
    background-color: #111827;
    border-right: 1px solid #273449;
}

.main-title {
    font-size: 36px;
    font-weight: 800;
    color: #F8FAFC;
    letter-spacing: -1px;
}

.subtitle {
    color: #94A3B8;
    font-size: 14px;
}

.section-title {
    color: #E2E8F0;
    font-size: 21px;
    font-weight: 700;
    margin-top: 15px;
}

.metric-card {
    background: linear-gradient(
        135deg,
        #17243A,
        #1E293B
    );
    border: 1px solid #334155;
    padding: 20px;
    border-radius: 12px;
}

.metric-label {
    color: #94A3B8;
    font-size: 13px;
}

.metric-value {
    color: #F8FAFC;
    font-size: 28px;
    font-weight: 800;
}

.prediction-high {
    background-color: #064E3B;
    color: #6EE7B7;
    padding: 15px;
    border-radius: 10px;
    font-weight: bold;
    text-align: center;
}

.prediction-regular {
    background-color: #713F12;
    color: #FDE68A;
    padding: 15px;
    border-radius: 10px;
    font-weight: bold;
    text-align: center;
}

div[data-testid="stMetric"] {
    background-color: #17243A;
    border: 1px solid #334155;
    padding: 15px;
    border-radius: 10px;
}

div[data-testid="stMetricLabel"] {
    color: #94A3B8;
}

</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------
# LOAD MODELS
# ------------------------------------------------------------

@st.cache_resource
def load_models():

    loaded = {}

    for name, config in MODEL_CONFIG.items():

        if config["file"].exists():

            loaded[name] = joblib.load(
                config["file"]
            )

    return loaded


models = load_models()


# ------------------------------------------------------------
# LOAD EVALUATION ARTIFACTS
# ------------------------------------------------------------

@st.cache_data
def load_evaluation_data():

    metrics_path = RESULTS_DIR / "model_comparison.csv"

    importance_path = RESULTS_DIR / "feature_importance.csv"

    metrics = (
        pd.read_csv(metrics_path)
        if metrics_path.exists()
        else pd.DataFrame()
    )

    importance = (
        pd.read_csv(importance_path)
        if importance_path.exists()
        else pd.DataFrame()
    )

    predictions = {}

    for name, config in MODEL_CONFIG.items():

        path = (
            RESULTS_DIR /
            f"predictions_{config['key']}.csv"
        )

        if path.exists():

            predictions[name] = pd.read_csv(path)

    return metrics, importance, predictions


metrics_df, importance_df, predictions_data = (
    load_evaluation_data()
)


# ------------------------------------------------------------
# COMMON FUNCTIONS
# ------------------------------------------------------------

def show_metric_card(label, value, suffix=""):

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}{suffix}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def predict_customer(features, selected_models, threshold):

    results = []

    for model_name, model in selected_models.items():

        expected_features = model.feature_names_in_

        X = features.reindex(
            columns=expected_features
        )

        probabilities = model.predict_proba(X)[0]

        classes = list(model.classes_)

        positive_index = classes.index(1)

        probability = probabilities[positive_index]

        prediction = int(probability >= threshold)

        results.append({
            "Model": model_name,
            "Prediction": prediction,
            "Probability": probability,
            "Classification": (
                "HIGH VALUE CUSTOMER"
                if prediction == 1
                else "REGULAR CUSTOMER"
            )
        })

    return pd.DataFrame(results)


def display_predictions(results):

    st.markdown(
        '<div class="section-title">Prediction Results</div>',
        unsafe_allow_html=True
    )

    columns = st.columns(len(results))

    for column, (_, result) in zip(
        columns,
        results.iterrows()
    ):

        with column:

            st.markdown(f"### {result['Model']}")

            st.metric(
                "High Value Probability",
                f"{result['Probability']:.2%}"
            )

            progress = float(result["Probability"])

            st.progress(progress)

            if result["Prediction"] == 1:

                st.markdown(
                    '<div class="prediction-high">'
                    '★ HIGH VALUE CUSTOMER'
                    '</div>',
                    unsafe_allow_html=True
                )

            else:

                st.markdown(
                    '<div class="prediction-regular">'
                    'REGULAR CUSTOMER'
                    '</div>',
                    unsafe_allow_html=True
                )

    st.markdown("")

    if results["Prediction"].nunique() == 1:

        st.success(
            "Both models agree on the customer classification."
        )

    else:

        st.warning(
            "The models disagree. Consider reviewing the "
            "customer profile and prediction probabilities."
        )


# ------------------------------------------------------------
# HEADER
# ------------------------------------------------------------

header_left, header_right = st.columns(
    [4, 1]
)

with header_left:

    st.markdown(
        '<div class="main-title">🎯 HighValue AI</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'CUSTOMER INTELLIGENCE PLATFORM | '
        'PREDICTIVE ANALYTICS'
        '</div>',
        unsafe_allow_html=True
    )

with header_right:

    st.markdown("")

    st.markdown(
        f"**{len(models)} Models Loaded**"
    )

st.divider()


# ------------------------------------------------------------
# SIDEBAR
# ------------------------------------------------------------

st.sidebar.markdown("# 🎯 HighValue AI")

st.sidebar.caption(
    "Customer Intelligence Platform"
)

page = st.sidebar.radio(
    "NAVIGATION",
    [
        "Customer Prediction",
        "Model Performance"
    ]
)

st.sidebar.divider()

st.sidebar.markdown("### Model Status")

for model_name in MODEL_CONFIG:

    if model_name in models:

        st.sidebar.success(
            f"{model_name}: Loaded"
        )

    else:

        st.sidebar.error(
            f"{model_name}: Missing"
        )

st.sidebar.divider()

threshold = st.sidebar.slider(
    "Classification Threshold",
    min_value=0.10,
    max_value=0.90,
    value=0.50,
    step=0.05
)

st.sidebar.caption(
    "Threshold used to classify a customer as High Value."
)


# ============================================================
# PAGE 1: CUSTOMER PREDICTION
# ============================================================

if page == "Customer Prediction":

    st.markdown(
        "## 🔮 Customer Prediction"
    )

    st.markdown(
        "Predict the future commercial value of a customer "
        "using historical purchase behavior."
    )

    if not models:

        st.error(
            "No trained models were found. "
            "Run train_model.py first."
        )

        st.stop()

    input_tab, upload_tab = st.tabs([
        "✍️ Manual Customer Input",
        "📂 Upload Transactions"
    ])

    # --------------------------------------------------------
    # MANUAL INPUT
    # --------------------------------------------------------

    with input_tab:

        st.markdown("### Customer Profile")

        st.caption(
            "Enter the historical features of a customer. "
            "Use the same observation date as the training dataset."
        )

        with st.form("manual_customer_form"):

            st.markdown("#### Purchase Behavior")

            c1, c2, c3 = st.columns(3)

            with c1:

                recency = st.number_input(
                    "Recency (days)",
                    min_value=0,
                    value=15
                )

                frequency = st.number_input(
                    "Frequency (orders)",
                    min_value=1,
                    value=10
                )

                monetary = st.number_input(
                    "Monetary (£)",
                    min_value=0.0,
                    value=1500.0
                )

                avg_order_value = st.number_input(
                    "Average Order Value (£)",
                    min_value=0.0,
                    value=150.0
                )

            with c2:

                total_quantity = st.number_input(
                    "Total Quantity",
                    min_value=0,
                    value=250
                )

                unique_products = st.number_input(
                    "Unique Products",
                    min_value=0,
                    value=30
                )

                avg_days = st.number_input(
                    "Average Days Between Orders",
                    min_value=0.0,
                    value=20.0
                )

                customer_tenure = st.number_input(
                    "Customer Tenure (days)",
                    min_value=0,
                    value=180
                )

            with c3:

                avg_items = st.number_input(
                    "Average Items Per Order",
                    min_value=0.0,
                    value=25.0
                )

                spend_30 = st.number_input(
                    "Spend Last 30 Days (£)",
                    min_value=0.0,
                    value=500.0
                )

                spend_60 = st.number_input(
                    "Spend Previous 60 Days (£)",
                    min_value=0.0,
                    value=700.0
                )

                country = st.selectbox(
                    "Country",
                    [
                        "United Kingdom",
                        "Germany",
                        "France",
                        "EIRE",
                        "Spain",
                        "Netherlands",
                        "Belgium",
                        "Portugal",
                        "Other"
                    ]
                )

            submitted = st.form_submit_button(
                "🚀 Predict Customer Value",
                use_container_width=True
            )

        if submitted:

            customer_features = pd.DataFrame([{
                "Recency": recency,
                "Frequency": frequency,
                "Monetary": monetary,
                "AvgOrderValue": avg_order_value,
                "TotalQuantity": total_quantity,
                "UniqueProducts": unique_products,
                "AvgDaysBetweenOrders": avg_days,
                "CustomerTenure": customer_tenure,
                "AvgItemsPerOrder": avg_items,
                "SpendLast30Days": spend_30,
                "SpendPrevious60Days": spend_60,
                "Country": country
            }])

            results = predict_customer(
                customer_features,
                models,
                threshold
            )

            st.session_state["last_results"] = results

            st.session_state["last_features"] = (
                customer_features
            )

            display_predictions(results)

            with st.expander("View Input Features"):

                st.dataframe(
                    customer_features,
                    use_container_width=True
                )

    # --------------------------------------------------------
    # UPLOAD TRANSACTIONS
    # --------------------------------------------------------

    with upload_tab:

        st.markdown("### Upload Customer Transactions")

        st.caption(
            "Upload a CSV or Excel file containing the "
            "historical transactions of exactly one customer."
        )

        uploaded_file = st.file_uploader(
            "Choose transaction file",
            type=["csv", "xlsx"]
        )

        if uploaded_file is not None:

            try:

                if uploaded_file.name.lower().endswith(".csv"):

                    raw_df = pd.read_csv(uploaded_file)

                else:

                    raw_df = pd.read_excel(uploaded_file)

                st.markdown("#### Uploaded Data")

                st.dataframe(
                    raw_df.head(10),
                    use_container_width=True
                )

                if st.button(
                    "🚀 Generate Customer Prediction",
                    use_container_width=True
                ):

                    cleaned_df = load_dataset(raw_df)

                    customers = cleaned_df[
                        "CustomerID"
                    ].unique()

                    if len(customers) != 1:

                        st.error(
                            "The uploaded file must contain "
                            "transactions from exactly one customer."
                        )

                    elif not (
                        cleaned_df["InvoiceDate"] <
                        REFERENCE_DATE
                    ).all():

                        st.error(
                            f"All transactions must be earlier "
                            f"than {REFERENCE_DATE.date()}."
                        )

                    else:

                        customer_features = (
                            build_customer_features(
                                cleaned_df,
                                REFERENCE_DATE
                            )
                        )

                        if customer_features.empty:

                            st.error(
                                "No valid customer features were generated."
                            )

                        else:

                            customer_id = (
                                customer_features.iloc[0]["CustomerID"]
                            )

                            X = customer_features.drop(
                                columns=["CustomerID"]
                            )

                            results = predict_customer(
                                X,
                                models,
                                threshold
                            )

                            st.success(
                                f"Customer ID: {customer_id}"
                            )

                            display_predictions(results)

                            st.markdown(
                                "#### Generated Features"
                            )

                            st.dataframe(
                                X,
                                use_container_width=True
                            )

            except Exception as error:

                st.error(
                    f"Unable to process the uploaded file: {error}"
                )


# ============================================================
# PAGE 2: MODEL PERFORMANCE
# ============================================================

elif page == "Model Performance":

    st.markdown(
        "## 📊 Model Performance"
    )

    st.markdown(
        "Analyze model quality, classification performance "
        "and the features that influence customer value prediction."
    )

    if metrics_df.empty:

        st.warning(
            "Evaluation metrics were not found. "
            "Execute train_model.py to generate the evaluation artifacts."
        )

        st.stop()

    # --------------------------------------------------------
    # OVERVIEW
    # --------------------------------------------------------

    st.markdown("### Model Comparison")

    st.caption(
        "Evaluation results generated during model training."
    )

    for metric in [
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "ROC_AUC",
        "PR_AUC"
    ]:

        if metric not in metrics_df.columns:
            st.warning(
                f"Metric {metric} not found in model_comparison.csv."
            )

    display_metrics = [
        col for col in [
            "Model",
            "Accuracy",
            "Precision",
            "Recall",
            "F1",
            "ROC_AUC",
            "PR_AUC"
        ]
        if col in metrics_df.columns
    ]

    st.dataframe(
        metrics_df[display_metrics].style.format({
            col: "{:.3f}"
            for col in display_metrics
            if col != "Model"
        }),
        use_container_width=True,
        hide_index=True
    )

    # --------------------------------------------------------
    # METRIC CHARTS
    # --------------------------------------------------------

    st.markdown("### Performance Metrics")

    metric_tabs = st.tabs([
        "Classification Metrics",
        "ROC-AUC",
        "Precision-Recall"
    ])

    with metric_tabs[0]:

        chart_data = metrics_df.melt(
            id_vars="Model",
            value_vars=[
                "Accuracy",
                "Precision",
                "Recall",
                "F1"
            ],
            var_name="Metric",
            value_name="Score"
        )

        fig = px.bar(
            chart_data,
            x="Metric",
            y="Score",
            color="Model",
            barmode="group",
            text_auto=".3f",
            title="Classification Metrics Comparison",
            color_discrete_map={
                "LogisticRegression": "#38BDF8",
                "RandomForest": "#A78BFA",
                "Logistic Regression": "#38BDF8",
                "Random Forest": "#A78BFA"
            }
        )

        fig.update_layout(
            template="plotly_dark",
            yaxis_range=[0, 1],
            paper_bgcolor="#0B1120",
            plot_bgcolor="#0B1120"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with metric_tabs[1]:

        st.markdown(
            "ROC curves show the trade-off between "
            "True Positive Rate and False Positive Rate."
        )

        fig = go.Figure()

        for name, pred_df in predictions_data.items():

            fpr, tpr, _ = roc_curve(
                pred_df["Actual"],
                pred_df["Probability"]
            )

            auc = roc_auc_score(
                pred_df["Actual"],
                pred_df["Probability"]
            )

            fig.add_trace(
                go.Scatter(
                    x=fpr,
                    y=tpr,
                    mode="lines",
                    name=f"{name} (AUC={auc:.3f})"
                )
            )

        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                name="Random Classifier",
                line=dict(dash="dash")
            )
        )

        fig.update_layout(
            title="ROC Curve",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            template="plotly_dark",
            paper_bgcolor="#0B1120",
            plot_bgcolor="#0B1120"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with metric_tabs[2]:

        fig = go.Figure()

        for name, pred_df in predictions_data.items():

            precision_values, recall_values, _ = (
                precision_recall_curve(
                    pred_df["Actual"],
                    pred_df["Probability"]
                )
            )

            ap = average_precision_score(
                pred_df["Actual"],
                pred_df["Probability"]
            )

            fig.add_trace(
                go.Scatter(
                    x=recall_values,
                    y=precision_values,
                    mode="lines",
                    name=f"{name} (AP={ap:.3f})"
                )
            )

        fig.update_layout(
            title="Precision-Recall Curve",
            xaxis_title="Recall",
            yaxis_title="Precision",
            template="plotly_dark",
            paper_bgcolor="#0B1120",
            plot_bgcolor="#0B1120"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.divider()

    # --------------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------------

    st.markdown("### Confusion Matrix")

    available_predictions = list(
        predictions_data.keys()
    )

    if available_predictions:

        selected_model = st.selectbox(
            "Select Model",
            available_predictions
        )

        pred_df = predictions_data[selected_model]

        cm = confusion_matrix(
            pred_df["Actual"],
            pred_df["Predicted"],
            labels=[0, 1]
        )

        fig = go.Figure(
            data=go.Heatmap(
                z=cm,
                x=["Predicted Regular", "Predicted High Value"],
                y=["Actual Regular", "Actual High Value"],
                text=cm,
                texttemplate="%{text}",
                colorscale="Blues",
                showscale=False
            )
        )

        fig.update_layout(
            title=f"{selected_model} - Confusion Matrix",
            template="plotly_dark",
            paper_bgcolor="#0B1120",
            plot_bgcolor="#0B1120"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        tn, fp, fn, tp = cm.ravel()

        c1, c2, c3, c4 = st.columns(4)

        c1.metric("True Negatives", tn)
        c2.metric("False Positives", fp)
        c3.metric("False Negatives", fn)
        c4.metric("True Positives", tp)

    else:

        st.info(
            "Prediction artifacts are not available."
        )

    st.divider()

    # --------------------------------------------------------
    # FEATURE IMPORTANCE
    # --------------------------------------------------------

    st.markdown("### Feature Importance")

    st.caption(
        "Random Forest feature importance generated during training."
    )

    if not importance_df.empty:

        top_n = st.slider(
            "Number of features",
            min_value=5,
            max_value=min(20, len(importance_df)),
            value=min(15, len(importance_df))
        )

        top_features = (
            importance_df
            .sort_values(
                "Importance",
                ascending=False
            )
            .head(top_n)
            .sort_values("Importance")
        )

        fig = px.bar(
            top_features,
            x="Importance",
            y="Feature",
            orientation="h",
            title="Most Important Features",
            color="Importance",
            color_continuous_scale="Purples"
        )

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0B1120",
            plot_bgcolor="#0B1120",
            showlegend=False
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.dataframe(
            importance_df.sort_values(
                "Importance",
                ascending=False
            ),
            use_container_width=True,
            hide_index=True
        )

    else:

        st.info(
            "Feature importance artifact not found."
        )

    st.divider()

    st.markdown("### Evaluation Dataset")

    st.caption(
        "The following metrics were calculated using the "
        "held-out test dataset."
    )

    if predictions_data:

        total_samples = len(
            next(iter(predictions_data.values()))
        )

        st.metric(
            "Test Samples",
            total_samples
        )