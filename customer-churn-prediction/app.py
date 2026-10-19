
import os
import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)


# ==========================================
# PROFESSIONAL UI STYLING
# ==========================================

st.markdown("""
<style>
/* Main application background */
.stApp {
    background-color: #F5F7FB;
}

/* Main content spacing */
.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1500px;
}

/* Main headings */
h1 {
    color: #14213D !important;
    font-weight: 800 !important;
    letter-spacing: -0.7px;
}

h2, h3 {
    color: #243B5A !important;
    font-weight: 700 !important;
}

/* Body text */
p, label, .stMarkdown {
    color: #334155;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #14213D;
}

[data-testid="stSidebar"] * {
    color: #F8FAFC !important;
}

/* Metric cards */
[data-testid="stMetric"] {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    padding: 20px;
    border-radius: 14px;
    box-shadow: 0 3px 12px rgba(15, 23, 42, 0.04);
}

/* Metric labels */
[data-testid="stMetricLabel"] {
    font-weight: 600;
}

/* Metric values */
[data-testid="stMetricValue"] {
    color: #2563EB;
    font-weight: 800;
}

/* Forms and bordered containers */
[data-testid="stForm"],
[data-testid="stFileUploader"] {
    background-color: #FFFFFF;
    padding: 20px;
    border: 1px solid #E2E8F0;
    border-radius: 14px;
}

/* Inputs */
div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div {
    border-radius: 8px;
}

/* Buttons */
.stButton > button,
.stFormSubmitButton > button,
.stDownloadButton > button {
    border-radius: 9px;
    font-weight: 650;
    min-height: 44px;
    transition: all 0.2s ease;
}

/* Primary action buttons */
.stFormSubmitButton > button[kind="primary"] {
    background-color: #2563EB;
    border: 1px solid #2563EB;
    color: white;
}

/* Button hover */
.stButton > button:hover,
.stFormSubmitButton > button:hover,
.stDownloadButton > button:hover {
    border-color: #1D4ED8;
    color: #1D4ED8;
}

/* Charts */
[data-testid="stPlotlyChart"] {
    background-color: #FFFFFF;
    border: 1px solid #E2E8F0;
    padding: 12px;
    border-radius: 14px;
}

/* Dataframes */
[data-testid="stDataFrame"] {
    border: 1px solid #E2E8F0;
    border-radius: 10px;
}

/* Dividers */
hr {
    border-color: #DCE3ED;
    margin-top: 1.5rem;
    margin-bottom: 1.5rem;
}

/* Status messages */
[data-testid="stAlert"] {
    border-radius: 10px;
}

/* Progress bars */
.stProgress > div > div > div {
    background-color: #2563EB;
}

/* Mobile adjustments */
@media (max-width: 768px) {
    .block-container {
        padding-left: 1rem;
        padding-right: 1rem;
    }
}
</style>
""", unsafe_allow_html=True)


MODEL_PATH = "models/churn_model.pkl"
COLUMNS_PATH = "models/model_columns.pkl"
DATA_PATH = "WA_Fn-UseC_-Telco-Customer-Churn.csv"

# ==========================================
# LOAD TRAINED MODEL
# ==========================================

if not os.path.exists(MODEL_PATH) or not os.path.exists(COLUMNS_PATH):
    st.error("Model files not found. Run train_model.py first.")
    st.stop()

model = joblib.load(MODEL_PATH)
model_columns = joblib.load(COLUMNS_PATH)

# Numeric features used by the Telco dataset
NUMERIC_COLUMNS = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges"
]


def prepare_input(data):
    """Prepare customer features for the saved ML pipeline."""
    data = data[model_columns].copy()

    for col in NUMERIC_COLUMNS:
        if col in data.columns:
            data[col] = pd.to_numeric(
                data[col], errors="coerce"
            )

    return data


# ==========================================
# HEADER AND SIDEBAR
# ==========================================

st.title("📊 Customer Churn Prediction System")
st.markdown(
    "Predict customer churn risk using a trained "
    "Machine Learning model."
)

st.sidebar.header("About This Project")
st.sidebar.write(
    "This application uses Logistic Regression trained "
    "on the Telco Customer Churn dataset."
)
st.sidebar.info(
    "Predictions are estimates based on historical data, "
    "not guarantees of future customer behavior."
)

# ==========================================
# INDIVIDUAL CUSTOMER PREDICTION
# ==========================================

st.divider()
st.header("👤 Individual Customer Prediction")

with st.form("customer_form"):

    st.subheader("Customer Profile")

    col1, col2, col3 = st.columns(3)

    with col1:
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior_citizen = st.selectbox(
            "Senior Citizen",
            [0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No"
        )
        partner = st.selectbox("Has Partner", ["Yes", "No"])
        dependents = st.selectbox(
            "Has Dependents", ["Yes", "No"]
        )

    with col2:
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        phone_service = st.selectbox(
            "Phone Service", ["Yes", "No"]
        )
        multiple_lines = st.selectbox(
            "Multiple Lines",
            ["No", "Yes", "No phone service"]
        )

    with col3:
        internet_service = st.selectbox(
            "Internet Service", ["DSL", "Fiber optic", "No"]
        )
        paperless_billing = st.selectbox(
            "Paperless Billing", ["Yes", "No"]
        )
        contract = st.selectbox(
            "Contract",
            ["Month-to-month", "One year", "Two year"]
        )

    st.subheader("Services and Billing")

    service_options = ["Yes", "No", "No internet service"]

    col4, col5, col6 = st.columns(3)

    with col4:
        online_security = st.selectbox(
            "Online Security", service_options
        )
        online_backup = st.selectbox(
            "Online Backup", service_options
        )
        device_protection = st.selectbox(
            "Device Protection", service_options
        )

    with col5:
        tech_support = st.selectbox(
            "Tech Support", service_options
        )
        streaming_tv = st.selectbox(
            "Streaming TV", service_options
        )
        streaming_movies = st.selectbox(
            "Streaming Movies", service_options
        )

    with col6:
        payment_method = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        )
        monthly_charges = st.number_input(
            "Monthly Charges ($)",
            min_value=0.0,
            max_value=1000.0,
            value=70.0,
            step=1.0
        )
        total_charges = st.number_input(
            "Total Charges ($)",
            min_value=0.0,
            max_value=100000.0,
            value=float(monthly_charges * tenure),
            step=10.0
        )

    submitted = st.form_submit_button(
        "Predict Customer Churn",
        type="primary",
        use_container_width=True
    )

if submitted:
    customer_data = {
        "gender": gender,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges
    }

    input_df = pd.DataFrame(
        [customer_data], columns=model_columns
    )
    input_df = prepare_input(input_df)

    try:
        prediction = int(model.predict(input_df)[0])
        probability = float(
            model.predict_proba(input_df)[0][1]
        )

        st.subheader("🔍 Prediction Results")
        result_col1, result_col2 = st.columns(2)

        with result_col1:
            if prediction == 1:
                st.error("⚠️ Customer Likely to Churn")
            else:
                st.success("✅ Customer Likely to Stay")

        with result_col2:
            st.metric(
                "Estimated Churn Probability",
                f"{probability:.1%}"
            )

        st.progress(probability)

        if probability >= 0.60:
            st.warning(
                "High estimated churn probability. "
                "Consider reviewing retention options."
            )
        elif probability >= 0.30:
            st.info(
                "Moderate estimated churn probability. "
                "Consider monitoring customer satisfaction."
            )
        else:
            st.success(
                "Low estimated churn probability "
                "according to the model."
            )

        st.caption(
            "The classification threshold is 0.50. "
            "Risk bands are illustrative, and actual outcomes "
            "may differ."
        )

    except Exception as error:
        st.error(f"Prediction failed: {error}")

# ==========================================
# CUSTOMER CHURN ANALYTICS DASHBOARD
# ==========================================

st.divider()
st.title("📊 Customer Churn Analytics Dashboard")
st.caption(
    "Explore insights from the original Telco customer dataset."
)

try:
    analytics_df = pd.read_csv(DATA_PATH)

    for col in ["TotalCharges", "tenure", "MonthlyCharges"]:
        analytics_df[col] = pd.to_numeric(
            analytics_df[col], errors="coerce"
        )

    total_customers = len(analytics_df)
    churned = (analytics_df["Churn"] == "Yes").sum()
    churn_rate = (
        churned / total_customers * 100
        if total_customers else 0
    )
    avg_monthly = analytics_df["MonthlyCharges"].mean()
    avg_tenure = analytics_df["tenure"].mean()

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Customers", f"{total_customers:,}")
    k2.metric("Customers Churned", f"{churned:,}")
    k3.metric("Churn Rate", f"{churn_rate:.1f}%")
    k4.metric(
        "Avg. Monthly Charges", f"${avg_monthly:,.2f}"
    )

    st.divider()

    left, right = st.columns(2)

    with left:
        st.subheader("Churn Distribution")

        churn_counts = (
            analytics_df["Churn"]
            .value_counts()
            .rename_axis("Churn")
            .reset_index(name="Customers")
        )

        fig_churn = px.pie(
            churn_counts,
            names="Churn",
            values="Customers",
            hole=0.55,
            color="Churn",
            color_discrete_map={
                "Yes": "#EF4444",
                "No": "#10B981"
            }
        )
        fig_churn.update_layout(
            legend_title_text="Customer Status"
        )
        st.plotly_chart(
            fig_churn, use_container_width=True
        )

    with right:
        st.subheader("Churn by Contract")

        contract_data = (
            analytics_df.groupby(["Contract", "Churn"])
            .size()
            .reset_index(name="Customers")
        )

        fig_contract = px.bar(
            contract_data,
            x="Contract",
            y="Customers",
            color="Churn",
            barmode="group",
            color_discrete_map={
                "Yes": "#EF4444",
                "No": "#10B981"
            }
        )
        st.plotly_chart(
            fig_contract, use_container_width=True
        )

    left, right = st.columns(2)

    with left:
        st.subheader("Customer Tenure Distribution")

        fig_tenure = px.histogram(
            analytics_df,
            x="tenure",
            color="Churn",
            nbins=24,
            barmode="overlay",
            opacity=0.75,
            color_discrete_map={
                "Yes": "#EF4444",
                "No": "#10B981"
            },
            labels={"tenure": "Tenure (months)"}
        )
        st.plotly_chart(
            fig_tenure, use_container_width=True
        )

    with right:
        st.subheader("Monthly Charges by Churn Status")

        fig_charges = px.box(
            analytics_df,
            x="Churn",
            y="MonthlyCharges",
            color="Churn",
            color_discrete_map={
                "Yes": "#EF4444",
                "No": "#10B981"
            },
            labels={
                "Churn": "Customer Churn",
                "MonthlyCharges": "Monthly Charges ($)"
            }
        )
        st.plotly_chart(
            fig_charges, use_container_width=True
        )

    st.subheader("Churn Rate by Contract Type")

    contract_rates = (
        analytics_df.groupby("Contract")
        .agg(
            Total_Customers=("Churn", "size"),
            Churned_Customers=(
                "Churn", lambda s: (s == "Yes").sum()
            )
        )
        .reset_index()
    )

    contract_rates["Churn Rate (%)"] = (
        contract_rates["Churned_Customers"]
        / contract_rates["Total_Customers"] * 100
    )

    st.dataframe(
        contract_rates.style.format(
            {"Churn Rate (%)": "{:.2f}%"}
        ),
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        f"Average customer tenure: {avg_tenure:.1f} months. "
        "These metrics describe the source dataset, "
        "not model predictions."
    )

except FileNotFoundError:
    st.warning(
        "Analytics dataset not found. Keep the CSV "
        "in the same folder as app.py."
    )
except (KeyError, ValueError) as error:
    st.error(f"Could not prepare dashboard data: {error}")

# ==========================================
# BULK CUSTOMER CHURN PREDICTION
# ==========================================

st.divider()
st.title("📁 Bulk Customer Churn Prediction")
st.write(
    "Upload a customer CSV file to predict churn "
    "for multiple customers at once."
)

uploaded_file = st.file_uploader(
    "Upload Customer CSV",
    type=["csv"],
    key="bulk_customer_csv"
)

if uploaded_file is not None:
    try:
        bulk_df = pd.read_csv(uploaded_file)

        if bulk_df.empty:
            st.error("The uploaded CSV contains no rows.")

        else:
            missing_columns = [
                col for col in model_columns
                if col not in bulk_df.columns
            ]

            if missing_columns:
                st.error(
                    "Missing required columns: "
                    + ", ".join(missing_columns)
                )
                st.info(
                    "Use the original Telco dataset as a template. "
                    "The Churn target column is optional."
                )

            else:
                # FIX: Convert numeric columns before prediction.
                # This prevents text values in TotalCharges from
                # reaching the numeric median imputer.
                prediction_input = prepare_input(bulk_df)

                # Reject rows missing critical numeric values
                # only if conversion made values missing? The
                # pipeline imputes missing values automatically.
                predictions = model.predict(prediction_input)
                probabilities = model.predict_proba(
                    prediction_input
                )[:, 1]

                results_df = bulk_df.copy()

                results_df["Predicted_Churn"] = [
                    "Yes" if int(p) == 1 else "No"
                    for p in predictions
                ]
                results_df["Churn_Probability"] = (
                    probabilities * 100
                ).round(2)

                st.success(
                    f"Successfully processed {len(results_df):,} customers."
                )

                total = len(results_df)
                predicted_churners = int(
                    (predictions == 1).sum()
                )
                predicted_rate = (
                    predicted_churners / total * 100
                    if total else 0
                )

                k1, k2, k3 = st.columns(3)
                k1.metric("Customers Processed", f"{total:,}")
                k2.metric(
                    "Predicted Churners",
                    f"{predicted_churners:,}"
                )
                k3.metric(
                    "Predicted Churn Rate",
                    f"{predicted_rate:.1f}%"
                )

                st.subheader("Prediction Results")
                st.dataframe(
                    results_df,
                    use_container_width=True,
                    hide_index=True
                )

                csv_data = results_df.to_csv(
                    index=False
                ).encode("utf-8")

                st.download_button(
                    label="⬇️ Download Prediction Results",
                    data=csv_data,
                    file_name="customer_churn_predictions.csv",
                    mime="text/csv",
                    use_container_width=True
                )

    except Exception as error:
        st.error(
            f"Unable to process this CSV: {error}"
        )
        st.info(
            "Check that the file is the Telco customer dataset "
            "and that all required feature columns are present."
        )


# ==========================================
# MODEL PERFORMANCE METRICS
# ==========================================

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)
from sklearn.model_selection import train_test_split

st.divider()
st.title("🎯 Model Performance Evaluation")
st.write(
    "Evaluate how well the trained Logistic Regression model "
    "identifies customers who are likely to churn."
)

PERFORMANCE_DATA_PATH = "WA_Fn-UseC_-Telco-Customer-Churn.csv"

try:
    performance_df = pd.read_csv(PERFORMANCE_DATA_PATH)

    # Prepare the dataset using the same target and features
    performance_df["TotalCharges"] = pd.to_numeric(
        performance_df["TotalCharges"],
        errors="coerce"
    )

    performance_df = performance_df.drop(
        columns=["customerID"],
        errors="ignore"
    )

    performance_df["Churn"] = (
        performance_df["Churn"].map({"Yes": 1, "No": 0})
    )

    X = performance_df.drop(columns=["Churn"])
    y = performance_df["Churn"]

    # Use the same split configuration as model training
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Evaluate the already-trained pipeline on the test split
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(
        y_test, y_pred, zero_division=0
    )
    recall = recall_score(
        y_test, y_pred, zero_division=0
    )
    f1 = f1_score(
        y_test, y_pred, zero_division=0
    )
    auc = roc_auc_score(y_test, y_prob)

    # Display key metrics
    st.subheader("📊 Key Performance Metrics")

    m1, m2, m3, m4 = st.columns(4)

    m1.metric("Accuracy", f"{accuracy:.1%}")
    m2.metric("Precision", f"{precision:.1%}")
    m3.metric("Recall", f"{recall:.1%}")
    m4.metric("ROC-AUC", f"{auc:.3f}")

    st.progress(accuracy, text="Accuracy")
    st.progress(precision, text="Precision")
    st.progress(recall, text="Recall")

    st.caption(
        f"F1 Score: {f1:.3f} | "
        f"Test samples: {len(y_test):,}"
    )

    # Confusion matrix
    st.subheader("🔎 Confusion Matrix")

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

    cm_df = pd.DataFrame(
        cm,
        index=["Actual: Stay", "Actual: Churn"],
        columns=["Predicted: Stay", "Predicted: Churn"]
    )

    st.dataframe(
        cm_df,
        use_container_width=True
    )

    tn, fp, fn, tp = cm.ravel()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Correctly Stayed", f"{tn:,}")
    c2.metric("False Churn Alerts", f"{fp:,}")
    c3.metric("Missed Churners", f"{fn:,}")
    c4.metric("Correctly Identified Churners", f"{tp:,}")

    st.info(
        "Recall is especially important for churn prevention: "
        "a low recall means the model misses some customers "
        "who actually leave. Precision helps measure how many "
        "churn alerts are correct."
    )

except FileNotFoundError:
    st.warning(
        "Performance dataset not found. Keep the original "
        "Telco CSV in the same folder as app.py."
    )
except Exception as error:
    st.error(f"Could not calculate model performance: {error}")
