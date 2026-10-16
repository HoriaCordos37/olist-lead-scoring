import io
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


st.set_page_config(
    page_title="Olist Lead Scoring & Seller Analytics",
    page_icon="🛒",
    layout="wide",
)

APP_DIR = Path(__file__).resolve().parent
REAL_PARQUET_PATH = APP_DIR / "data" / "seller_marketing_performance.parquet"
REAL_CSV_PATH = APP_DIR / "data" / "seller_marketing_performance.csv"
DEMO_CSV_PATH = APP_DIR / "data" / "demo_seller_marketing_performance.csv"

REVENUE_THRESHOLD_FROM_NOTEBOOK = 3289.80

CATEGORICAL_FEATURES = [
    "lead_origin",
    "business_segment",
    "lead_type",
    "lead_behaviour_profile",
    "has_company",
    "has_gtin",
    "average_stock",
    "business_type",
    "seller_state",
]

NUMERIC_FEATURES = [
    "declared_monthly_revenue",
    "days_to_close",
]

REQUIRED_MODEL_COLUMNS = CATEGORICAL_FEATURES + NUMERIC_FEATURES + ["high_value_seller"]


def section_title(title: str, subtitle: str | None = None) -> None:
    st.markdown(f"## {title}")
    if subtitle:
        st.caption(subtitle)


def read_uploaded_file(uploaded_file) -> pd.DataFrame:
    filename = uploaded_file.name.lower()
    if filename.endswith(".parquet"):
        return pd.read_parquet(uploaded_file)
    if filename.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    raise ValueError("Please upload a CSV or Parquet file.")


@st.cache_data(show_spinner=False)
def load_local_data() -> tuple[pd.DataFrame, str]:
    if REAL_PARQUET_PATH.exists():
        return pd.read_parquet(REAL_PARQUET_PATH), "real parquet"
    if REAL_CSV_PATH.exists():
        return pd.read_csv(REAL_CSV_PATH), "real csv"
    if DEMO_CSV_PATH.exists():
        return pd.read_csv(DEMO_CSV_PATH), "demo sample"
    raise FileNotFoundError(
        "No dataset found. Put seller_marketing_performance.parquet or .csv in the data folder."
    )


def normalize_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Create the target if the exported file contains revenue but not the label.
    if "high_value_seller" not in df.columns and "total_revenue" in df.columns:
        df["high_value_seller"] = (
            df["total_revenue"] >= REVENUE_THRESHOLD_FROM_NOTEBOOK
        ).astype(int)

    for col in CATEGORICAL_FEATURES:
        if col not in df.columns:
            df[col] = "unknown"
        df[col] = df[col].fillna("unknown").astype(str)

    for col in NUMERIC_FEATURES:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if "high_value_seller" in df.columns:
        df["high_value_seller"] = pd.to_numeric(
            df["high_value_seller"], errors="coerce"
        ).fillna(0).astype(int)

    numeric_report_cols = [
        "total_revenue",
        "total_orders",
        "customers_reached",
        "different_products_sold",
        "average_item_price",
        "average_freight_value",
        "average_review_score",
        "average_delivery_time_days",
        "average_delivery_delay_days",
        "declared_product_catalog_size",
    ]
    for col in numeric_report_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def can_train_model(df: pd.DataFrame) -> tuple[bool, str]:
    missing = [col for col in REQUIRED_MODEL_COLUMNS if col not in df.columns]
    if missing:
        return False, f"Missing columns: {', '.join(missing)}"

    if df["high_value_seller"].nunique() < 2:
        return False, "The target column has only one class."

    return True, "Ready"


@st.cache_resource(show_spinner=False)
def train_model(df: pd.DataFrame):
    model_df = df[REQUIRED_MODEL_COLUMNS].copy()
    X = model_df[CATEGORICAL_FEATURES + NUMERIC_FEATURES]
    y = model_df["high_value_seller"].astype(int)

    test_size = 0.25 if len(model_df) >= 80 else 0.30

    stratify = y if y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=37,
        stratify=stratify,
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", categorical_transformer, CATEGORICAL_FEATURES),
            ("numeric", numeric_transformer, NUMERIC_FEATURES),
        ]
    )

    classifier = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        random_state=49,
        class_weight="balanced",
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", classifier),
        ]
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1-score": f1_score(y_test, y_pred, zero_division=0),
    }

    if hasattr(model, "predict_proba"):
        try:
            y_proba = model.predict_proba(X_test)[:, 1]
            metrics["ROC-AUC"] = roc_auc_score(y_test, y_proba)
        except Exception:
            metrics["ROC-AUC"] = np.nan

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

    return model, metrics, cm


def get_feature_importance(model: Pipeline) -> pd.DataFrame:
    preprocessor = model.named_steps["preprocessor"]
    rf_model = model.named_steps["model"]

    onehot = preprocessor.named_transformers_["categorical"].named_steps["onehot"]
    categorical_names = onehot.get_feature_names_out(CATEGORICAL_FEATURES).tolist()
    feature_names = categorical_names + NUMERIC_FEATURES

    importances = rf_model.feature_importances_

    return (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .head(15)
    )


def money(value) -> str:
    if pd.isna(value):
        return "N/A"
    return f"R$ {value:,.0f}"


def pct(value) -> str:
    if pd.isna(value):
        return "N/A"
    return f"{value:.1%}"


def unique_options(df: pd.DataFrame, column: str) -> list[str]:
    values = df[column].dropna().astype(str).unique().tolist()
    return sorted(values) if values else ["unknown"]


def safe_default_numeric(df: pd.DataFrame, column: str, fallback: float) -> float:
    if column not in df.columns:
        return fallback
    value = pd.to_numeric(df[column], errors="coerce").median()
    if pd.isna(value):
        return fallback
    return float(value)


# Sidebar data loading
st.sidebar.title("🛒 Olist Advertising Demo")
uploaded_file = st.sidebar.file_uploader(
    "Optional: upload your real final dataset",
    type=["csv", "parquet"],
    help="Upload seller_marketing_performance.parquet or seller_marketing_performance.csv from Colab.",
)

try:
    if uploaded_file is not None:
        data = normalize_dataset(read_uploaded_file(uploaded_file))
        data_source = f"uploaded file: {uploaded_file.name}"
    else:
        local_df, local_source = load_local_data()
        data = normalize_dataset(local_df)
        data_source = local_source
except Exception as exc:
    st.error(f"Could not load the dataset: {exc}")
    st.stop()

can_train, train_message = can_train_model(data)
model = None
metrics = {}
cm = np.array([[0, 0], [0, 0]])

if can_train:
    model, metrics, cm = train_model(data)

if data_source == "demo sample":
    st.sidebar.warning(
        "Using included demo sample data. Replace it with your real Colab export for the final presentation."
    )
else:
    st.sidebar.success(f"Using {data_source}")

page = st.sidebar.radio(
    "Choose page",
    [
        "1. Project Overview",
        "2. Data Explorer",
        "3. Business Dashboard",
        "4. Prediction Demo",
        "5. Model Performance",
    ],
)

st.sidebar.markdown("---")
st.sidebar.caption("Project goal: predict whether a marketing lead becomes a high-value seller.")


if page == "1. Project Overview":
    st.title("Olist Lead Scoring & Seller Analytics")
    st.caption("Big Data + Machine Learning demo based on the Olist e-commerce and marketing funnel datasets.")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Final dataset rows", f"{len(data):,}")
    col2.metric("Final dataset columns", f"{data.shape[1]:,}")
    col3.metric(
        "High-value sellers",
        f"{int(data['high_value_seller'].sum()):,}" if "high_value_seller" in data.columns else "N/A",
    )
    col4.metric(
        "High-value rate",
        pct(data["high_value_seller"].mean()) if "high_value_seller" in data.columns else "N/A",
    )

    section_title("Business problem")
    st.write(
        "The app predicts whether a marketing lead or seller is likely to become a high-value e-commerce seller. "
        "This helps a marketplace prioritize sales follow-up, advertising budget, and seller acquisition effort."
    )

    section_title("Project pipeline")
    st.code(
        """Kaggle Olist datasets
    → raw CSV storage
    → PySpark cleaning and joins
    → processed Parquet files
    → final seller_marketing_performance dataset
    → Random Forest prediction app
""",
        language="text",
    )

    section_title("Actual output from the notebook")
    st.markdown(
        """
- Raw storage: 9 Olist e-commerce CSV files + 2 marketing funnel CSV files.
- Processed storage: cleaned e-commerce Parquet + cleaned marketing Parquet.
- Final integrated output: `seller_marketing_performance.parquet`.
- Prediction target: `high_value_seller`.
- Notebook threshold used for high-value sellers: revenue ≥ `3289.80`.
"""
    )

    if data_source == "demo sample":
        st.info(
            "This package includes sample data so the Streamlit app runs immediately. "
            "For your real demo, export the final dataset from Colab and place it in the data folder."
        )


elif page == "2. Data Explorer":
    st.title("Data Explorer")
    st.caption(f"Current data source: {data_source}")

    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", f"{data.shape[0]:,}")
    col2.metric("Columns", f"{data.shape[1]:,}")
    col3.metric("Missing values", f"{int(data.isna().sum().sum()):,}")

    section_title("Dataset preview")
    st.dataframe(data.head(50), use_container_width=True)

    section_title("Column list")
    col_a, col_b = st.columns([1, 2])
    with col_a:
        selected_column = st.selectbox("Inspect one column", data.columns)
    with col_b:
        st.write(data[selected_column].describe(include="all"))

    section_title("Download current dataset")
    csv_buffer = io.StringIO()
    data.to_csv(csv_buffer, index=False)
    st.download_button(
        "Download current data as CSV",
        data=csv_buffer.getvalue(),
        file_name="seller_marketing_performance_current.csv",
        mime="text/csv",
    )


elif page == "3. Business Dashboard":
    st.title("Business Dashboard")
    st.caption("Charts that connect advertising channels, business segments, and seller quality.")

    if "total_revenue" in data.columns:
        total_revenue = data["total_revenue"].sum()
        avg_revenue = data["total_revenue"].mean()
    else:
        total_revenue = np.nan
        avg_revenue = np.nan

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total seller revenue", money(total_revenue))
    col2.metric("Average seller revenue", money(avg_revenue))
    col3.metric("High-value rate", pct(data["high_value_seller"].mean()))
    col4.metric("Revenue threshold", money(REVENUE_THRESHOLD_FROM_NOTEBOOK))

    left, right = st.columns(2)

    with left:
        target_counts = (
            data["high_value_seller"]
            .map({0: "Normal seller", 1: "High-value seller"})
            .value_counts()
            .reset_index()
        )
        target_counts.columns = ["seller_class", "count"]
        fig = px.bar(
            target_counts,
            x="seller_class",
            y="count",
            title="Seller Class Distribution",
            text_auto=True,
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        if "lead_origin" in data.columns:
            quality_by_origin = (
                data.groupby("lead_origin", dropna=False)
                .agg(
                    high_value_rate=("high_value_seller", "mean"),
                    seller_count=("high_value_seller", "size"),
                )
                .query("seller_count >= 3")
                .sort_values("high_value_rate", ascending=False)
                .reset_index()
            )
            fig = px.bar(
                quality_by_origin,
                x="high_value_rate",
                y="lead_origin",
                orientation="h",
                title="High-value Seller Rate by Lead Origin",
                text=quality_by_origin["high_value_rate"].map(lambda x: f"{x:.0%}"),
            )
            fig.update_layout(yaxis={"categoryorder": "total ascending"})
            fig.update_xaxes(tickformat=".0%")
            st.plotly_chart(fig, use_container_width=True)

    if "business_segment" in data.columns and "total_revenue" in data.columns:
        revenue_by_segment = (
            data.groupby("business_segment", dropna=False)
            .agg(
                average_total_revenue=("total_revenue", "mean"),
                seller_count=("total_revenue", "size"),
            )
            .query("seller_count >= 3")
            .sort_values("average_total_revenue", ascending=False)
            .head(12)
            .reset_index()
        )
        fig = px.bar(
            revenue_by_segment,
            x="average_total_revenue",
            y="business_segment",
            orientation="h",
            title="Top Business Segments by Average Seller Revenue",
            text=revenue_by_segment["average_total_revenue"].map(lambda x: f"R$ {x:,.0f}"),
        )
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)

    if "seller_state" in data.columns:
        state_quality = (
            data.groupby("seller_state", dropna=False)
            .agg(
                sellers=("high_value_seller", "size"),
                high_value_rate=("high_value_seller", "mean"),
            )
            .sort_values("sellers", ascending=False)
            .reset_index()
        )
        fig = px.scatter(
            state_quality,
            x="sellers",
            y="high_value_rate",
            size="sellers",
            hover_name="seller_state",
            title="Seller Volume and High-value Rate by State",
        )
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)


elif page == "4. Prediction Demo":
    st.title("Prediction Demo")
    st.caption("Enter a lead profile and the app predicts whether the seller may become high-value.")

    if not can_train or model is None:
        st.error(f"The model cannot be trained: {train_message}")
        st.stop()

    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            lead_origin = st.selectbox("Lead origin", unique_options(data, "lead_origin"))
            business_segment = st.selectbox("Business segment", unique_options(data, "business_segment"))
            lead_type = st.selectbox("Lead type", unique_options(data, "lead_type"))

        with col2:
            lead_behaviour_profile = st.selectbox(
                "Lead behaviour profile", unique_options(data, "lead_behaviour_profile")
            )
            has_company = st.selectbox("Has company?", unique_options(data, "has_company"))
            has_gtin = st.selectbox("Has GTIN?", unique_options(data, "has_gtin"))

        with col3:
            average_stock = st.selectbox("Average stock", unique_options(data, "average_stock"))
            business_type = st.selectbox("Business type", unique_options(data, "business_type"))
            seller_state = st.selectbox("Seller state", unique_options(data, "seller_state"))

        col4, col5 = st.columns(2)
        with col4:
            declared_monthly_revenue = st.number_input(
                "Declared monthly revenue",
                min_value=0.0,
                value=safe_default_numeric(data, "declared_monthly_revenue", 3000.0),
                step=500.0,
            )
        with col5:
            days_to_close = st.number_input(
                "Days to close",
                min_value=0.0,
                value=safe_default_numeric(data, "days_to_close", 30.0),
                step=1.0,
            )

        submitted = st.form_submit_button("Predict seller quality")

    if submitted:
        input_df = pd.DataFrame(
            [
                {
                    "lead_origin": lead_origin,
                    "business_segment": business_segment,
                    "lead_type": lead_type,
                    "lead_behaviour_profile": lead_behaviour_profile,
                    "has_company": has_company,
                    "has_gtin": has_gtin,
                    "average_stock": average_stock,
                    "business_type": business_type,
                    "seller_state": seller_state,
                    "declared_monthly_revenue": declared_monthly_revenue,
                    "days_to_close": days_to_close,
                }
            ]
        )

        probability = float(model.predict_proba(input_df)[0, 1])
        prediction = int(probability >= 0.50)

        if prediction == 1:
            st.success(f"Prediction: High-value seller | Confidence: {probability:.1%}")
            st.write(
                "Suggested action: prioritize this lead for sales follow-up, onboarding support, and advertising investment."
            )
        else:
            st.warning(f"Prediction: Normal-value seller | High-value probability: {probability:.1%}")
            st.write(
                "Suggested action: keep the lead in the nurturing flow and collect more information before assigning premium sales effort."
            )

        st.progress(min(max(probability, 0), 1), text=f"High-value probability: {probability:.1%}")

    st.info(
        "This prediction is for demo and decision-support only. It should not replace human review."
    )


elif page == "5. Model Performance":
    st.title("Model Performance")
    st.caption("The app trains a lightweight Random Forest model from the current dataset.")

    if not can_train or model is None:
        st.error(f"The model cannot be trained: {train_message}")
        st.stop()

    metric_cols = st.columns(len(metrics))
    for col, (name, value) in zip(metric_cols, metrics.items()):
        col.metric(name, "N/A" if pd.isna(value) else f"{value:.3f}")

    section_title("Confusion matrix")
    cm_df = pd.DataFrame(
        cm,
        index=["Actual normal", "Actual high-value"],
        columns=["Predicted normal", "Predicted high-value"],
    )
    st.dataframe(cm_df, use_container_width=True)

    fig = px.imshow(
        cm_df,
        text_auto=True,
        aspect="auto",
        title="Random Forest Confusion Matrix",
    )
    st.plotly_chart(fig, use_container_width=True)

    section_title("Top model features")
    importance_df = get_feature_importance(model)
    fig = px.bar(
        importance_df.sort_values("importance"),
        x="importance",
        y="feature",
        orientation="h",
        title="Top Feature Importances",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(importance_df, use_container_width=True)
