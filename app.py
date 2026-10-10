
from pathlib import Path

import streamlit as st
import pandas as pd
import joblib
from datetime import datetime
import plotly.graph_objects as go

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "logistic_regression.joblib"
DATA_PATH = BASE_DIR / "data" / "prepared_credit_data.csv"


st.set_page_config(
    page_title="Credit Scoring AI",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)




st.markdown("""
<style>
    .stApp {
        background-color: #0b1120;
        color: #e5e7eb;
    }

    [data-testid="stHeader"] {
        background-color: #0b1120;
    }

    [data-testid="stSidebar"] {
        background-color: #111b2e;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1400px;
    }

    h1, h2, h3 {
        color: #f8fafc;
    }

    div[data-testid="stMetric"] {
        background-color: #162238;
        border: 1px solid #263752;
        padding: 18px;
        border-radius: 14px;
    }

    div[data-testid="stForm"] {
        background-color: #111b2e;
        border: 1px solid #263752;
        padding: 20px;
        border-radius: 16px;
    }

    div.stButton > button,
    div[data-testid="stFormSubmitButton"] > button {
        background-color: #2563eb;
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.65rem 1rem;
        font-weight: 600;
        min-height: 45px;
    }

    div.stButton > button:hover,
    div[data-testid="stFormSubmitButton"] > button:hover {
        background-color: #1d4ed8;
        color: white;
        border: 1px solid #60a5fa;
    }

    [data-testid="stAlert"] {
        border-radius: 12px;
    }

    hr {
        border-color: #263752;
    }
</style>
""", unsafe_allow_html=True)



@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


model = load_model()
df = load_data()
if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []

st.sidebar.download_button(
    label="📥 Download Credit Dataset",
    data=df.to_csv(index=False).encode("utf-8"),
    file_name="prepared_credit_data.csv",
    mime="text/csv",
)



# Translate dataset codes into readable labels.
LABELS = {
    "checking_status": {
        "A11": "Below 0 DM",
        "A12": "0 to 200 DM",
        "A13": "200 DM or more",
        "A14": "No checking account",
    },
    "credit_history": {
        "A30": "No credits / all paid back duly",
        "A31": "All credits at this bank paid back duly",
        "A32": "Existing credits paid back duly",
        "A33": "Delay in paying past credits",
        "A34": "Critical account / other credits elsewhere",
    },
    "purpose": {
        "A40": "Car (new)",
        "A41": "Car (used)",
        "A42": "Furniture / equipment",
        "A43": "Radio / television",
        "A44": "Domestic appliances",
        "A45": "Repairs",
        "A46": "Education",
        "A47": "Vacation",
        "A48": "Retraining",
        "A49": "Business",
        "A410": "Others",
    },
    "savings": {
        "A61": "Below 100 DM",
        "A62": "100 to 500 DM",
        "A63": "500 to 1000 DM",
        "A64": "1000 DM or more",
        "A65": "Unknown / no savings account",
    },
    "employment_since": {
        "A71": "Unemployed",
        "A72": "Less than 1 year",
        "A73": "1 to 4 years",
        "A74": "4 to 7 years",
        "A75": "7 years or more",
    },
    "personal_status_sex": {
        "A91": "Male: divorced / separated",
        "A92": "Female: divorced / separated / married",
        "A93": "Male: single",
        "A94": "Male: married / widowed",
        "A95": "Female: single",
    },
    "other_debtors": {
        "A101": "None",
        "A102": "Co-applicant",
        "A103": "Guarantor",
    },
    "property": {
        "A121": "Real estate",
        "A122": "Building savings / life insurance",
        "A123": "Car or other property",
        "A124": "Unknown / no property",
    },
    "other_installment_plans": {
        "A141": "Bank",
        "A142": "Stores",
        "A143": "None",
    },
    "housing": {
        "A151": "Rent",
        "A152": "Own",
        "A153": "Free housing",
    },
    "job": {
        "A171": "Unemployed / unskilled non-resident",
        "A172": "Unskilled resident",
        "A173": "Skilled employee / official",
        "A174": "Management / self-employed / highly qualified",
    },
    "telephone": {
        "A191": "No telephone",
        "A192": "Telephone registered in applicant's name",
    },
    "foreign_worker": {
        "A201": "Yes",
        "A202": "No",
    },
}


def make_dropdown(column, label):
    options = sorted(df[column].dropna().unique())

    return st.selectbox(
        label,
        options,
        format_func=lambda code: LABELS.get(
            column, {}
        ).get(code, code),
    )



# Dashboard header
st.markdown("""
<div style="
    background: linear-gradient(120deg, #172554, #1e40af, #0f766e);
    padding: 28px;
    border-radius: 18px;
    margin-bottom: 24px;
">
    <div style="font-size: 14px; color: #bfdbfe;">
        AI-POWERED FINANCIAL ANALYTICS
    </div>
    <h1 style="color: white; margin: 8px 0;">
        💳 Credit Scoring Dashboard
    </h1>
    <p style="color: #e0f2fe; margin-bottom: 0;">
        Evaluate applicant information and estimate credit risk
        using machine learning.
    </p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.title("💳 CreditScore AI")
    st.caption("Machine Learning Dashboard")
    st.divider()

    page = st.radio(
        "Navigation",
        ["🏠 Credit Prediction", "ℹ️ About Project"],
    )

    st.divider()
    st.markdown("### Model Information")
    st.write("Algorithm: Logistic Regression")
    st.write("Dataset: German Credit")
    st.write("Features: Applicant financial profile")

    st.caption("For educational use only.")



st.info(
    "Educational demonstration only. Predictions may be inaccurate "
    "and should not be used alone for actual lending decisions."
)

if page == "🏠 Credit Prediction":
    st.subheader("Enter Applicant Details")


# IMPORTANT: All input fields and the submit button
# must be inside this single form.
with st.form("credit_form"):

    col1, col2 = st.columns(2)

    with col1:
        checking_status = make_dropdown(
            "checking_status", "Checking Account Status"
        )

        duration = st.number_input(
            "Loan Duration (months)",
            min_value=1, max_value=100, value=12,
        )

        credit_history = make_dropdown(
            "credit_history", "Credit History"
        )

        purpose = make_dropdown(
            "purpose", "Loan Purpose"
        )

        credit_amount = st.number_input(
            "Credit Amount",
            min_value=1, max_value=100000, value=2000,
        )

        savings = make_dropdown(
            "savings", "Savings Status"
        )

        employment_since = make_dropdown(
            "employment_since", "Employment Duration"
        )

        installment_rate = st.slider(
            "Installment Rate (1–4)", 1, 4, 2
        )

        personal_status_sex = make_dropdown(
            "personal_status_sex", "Personal Status"
        )

        other_debtors = make_dropdown(
            "other_debtors", "Other Debtors"
        )

        residence_since = st.slider(
            "Years at Current Residence", 1, 4, 2
        )

    with col2:
        property_value = make_dropdown(
            "property", "Property"
        )

        age = st.number_input(
            "Age", min_value=18, max_value=100, value=30
        )

        other_installment_plans = make_dropdown(
            "other_installment_plans",
            "Other Installment Plans",
        )

        housing = make_dropdown(
            "housing", "Housing"
        )

        existing_credits = st.number_input(
            "Existing Credits", min_value=1, max_value=10, value=1
        )

        job = make_dropdown(
            "job", "Job Category"
        )

        dependents = st.number_input(
            "Number of Dependents", min_value=1, max_value=10, value=1
        )

        telephone = make_dropdown(
            "telephone", "Telephone"
        )

        foreign_worker = make_dropdown(
            "foreign_worker", "Foreign Worker"
        )

    submitted = st.form_submit_button(
        "Predict Credit Risk",
        use_container_width=True,
    )


 
# Prediction code
if submitted:
    monthly_burden = credit_amount / duration

    applicant = pd.DataFrame([{
        "checking_status": checking_status,
        "duration": duration,
        "credit_history": credit_history,
        "purpose": purpose,
        "credit_amount": credit_amount,
        "savings": savings,
        "employment_since": employment_since,
        "installment_rate": installment_rate,
        "personal_status_sex": personal_status_sex,
        "other_debtors": other_debtors,
        "residence_since": residence_since,
        "property": property_value,
        "age": age,
        "other_installment_plans": other_installment_plans,
        "housing": housing,
        "existing_credits": existing_credits,
        "job": job,
        "dependents": dependents,
        "telephone": telephone,
        "foreign_worker": foreign_worker,
        "monthly_burden": monthly_burden,
    }])

    prediction = int(model.predict(applicant)[0])
    probabilities = model.predict_proba(applicant)[0]
    bad_probability = float(probabilities[1])

    st.session_state.prediction_history.append({
        "Time": datetime.now().strftime("%H:%M:%S"),
        "Predicted Risk": (
            "Bad Credit Risk" if prediction == 1
            else "Good Credit Risk"
        ),
        "Bad-Credit Probability": f"{bad_probability:.1%}",
    })

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("Model Prediction: Bad Credit Risk")
    else:
        st.success("Model Prediction: Good Credit Risk")

    result_col1, result_col2 = st.columns(2)

    result_col1.metric(
        "Estimated Good-Credit Probability",
        f"{probabilities[0]:.1%}",
    )

    result_col2.metric(
        "Estimated Bad-Credit Probability",
        f"{bad_probability:.1%}",
    )

    st.subheader("📊 Bad-Credit Risk Gauge")

    gauge = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=bad_probability * 100,
            number={"suffix": "%", "valueformat": ".1f"},
            title={"text": "Estimated Bad-Credit Probability"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#6366F1"},
                "steps": [
                    {"range": [0, 30], "color": "#D1FAE5"},
                    {"range": [30, 60], "color": "#FEF3C7"},
                    {"range": [60, 100], "color": "#FEE2E2"},
                ],
            },
        )
    )

    gauge.update_layout(
        height=300,
        margin=dict(l=25, r=25, t=65, b=15),
    )

    st.plotly_chart(gauge, use_container_width=True)

    if bad_probability < 0.30:
        st.info(
            "The model estimates a relatively lower bad-credit probability."
        )
    elif bad_probability < 0.60:
        st.warning(
            "The model estimates a moderate bad-credit probability."
        )
    else:
        st.error(
            "The model estimates a higher bad-credit probability."
        )

    report = pd.DataFrame([{
        "Predicted Risk": (
            "Bad Credit Risk" if prediction == 1
            else "Good Credit Risk"
        ),
        "Good-Credit Probability": round(float(probabilities[0]), 4),
        "Bad-Credit Probability": round(float(probabilities[1]), 4),
    }])

    st.download_button(
        label="📥 Download Prediction Report",
        data=report.to_csv(index=False).encode("utf-8"),
        file_name="credit_prediction_report.csv",
        mime="text/csv",
    )

    st.markdown("### Credit Risk Indicator")
    st.progress(bad_probability)
    st.caption(
        f"Estimated bad-credit probability: {bad_probability:.1%}"
    )


    
# Model performance section
st.divider()
st.subheader("📊 Model Performance Overview")

st.markdown("### 📋 Detailed Model Comparison")

comparison = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Random Forest",
        "Decision Tree"
    ],
    "Accuracy": ["75.5%", "76.5%", "66.0%"],
    "Precision": ["56.3%", "61.0%", "44.9%"],
    "Recall": ["81.7%", "60.0%", "58.3%"],
    "F1 Score": ["66.7%", "60.5%", "50.7%"],
    "ROC-AUC": ["80.7%", "79.0%", "61.3%"]
})

st.dataframe(
    comparison,
    use_container_width=True,
    hide_index=True
)

comparison["Accuracy_numeric"] = pd.to_numeric(
    comparison["Accuracy"].astype(str).str.replace("%", "", regex=False),
    errors="coerce"
)

if comparison["Accuracy_numeric"].notna().any():
    best_model = comparison.loc[
        comparison["Accuracy_numeric"].idxmax()
    ]

    st.success(
        f"🏆 Best model by accuracy: {best_model.iloc[0]} "
        f"({best_model['Accuracy_numeric']:.1f}%)"
    )
else:
    st.warning("Model accuracy values are unavailable.")

st.caption(
    "💡 Model recommendation is based on test accuracy only. "
    "Accuracy alone does not determine the best model for every use case."
)

st.info(
    "Important: This model provides an estimated credit-risk "
    "prediction for educational and decision-support purposes. "
    "It should not be used as the sole basis for loan approval "
    "or rejection."
)



metric1, metric2, metric3 = st.columns(3)

metric1.metric(
    "Logistic Regression Accuracy",
    "75.5%",
)

metric2.metric(
    "Random Forest Accuracy",
    "76.5%",
)

metric3.metric(
    "Decision Tree Accuracy",
    "66.0%",
)

st.markdown("### ROC-AUC Comparison")

performance = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Random Forest",
        "Decision Tree",
    ],
    "ROC-AUC": [0.806905, 0.790060, 0.612738],
})

st.bar_chart(
    performance.set_index("Model"),
    y="ROC-AUC",
)

st.caption(
    "Performance values come from the current test-set evaluation. "
    "Results may differ with different data splits."
)


st.caption(
        "These probabilities are model estimates, not guarantees. "
        "The application is for educational use, not actual loan approval."
    )

st.divider()
st.subheader("📊 Credit Risk Distribution")

risk_counts = df["class"].map({
    0: "Good Credit",
    1: "Bad Credit"
}).value_counts()

chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    st.markdown("#### Credit Categories")
    st.bar_chart(risk_counts)

with chart_col2:
    st.markdown("#### Dataset Summary")
    good_count = int((df["class"] == 0).sum())
    bad_count = int((df["class"] == 1).sum())

    st.metric("Good Credit Records", good_count)
    st.metric("Bad Credit Records", bad_count)

    st.caption(
        "Distribution of labels in the prepared German Credit dataset."
    )



if page == "ℹ️ About Project":
    st.title("ℹ️ About Credit Scoring AI")

    st.write(
        "Credit Scoring AI is a machine learning project designed "
        "to estimate the credit risk of loan applicants using their "
        "financial and personal information."
    )

    st.subheader("🎯 Project Objectives")
    st.write("- Estimate the likelihood of bad credit.")
    st.write("- Compare multiple machine learning algorithms.")
    st.write("- Present predictions through an interactive dashboard.")

    st.subheader("🤖 Machine Learning Models")
    st.write("1. Logistic Regression")
    st.write("2. Random Forest Classifier")
    st.write("3. Decision Tree Classifier")

    st.subheader("📊 Dataset")
    st.write(
        "The project uses the German Credit dataset, containing "
        "applicant information and credit classifications."
    )

    st.subheader("🛠️ Technologies Used")
    st.write(
        "Python, Pandas, Scikit-learn, Joblib, and Streamlit."
    )

    st.warning(
        "This project is for educational purposes. Its predictions "
        "should not be used as the sole basis for real loan decisions."
    )
st.divider()
st.subheader("🕒 Prediction History")

if st.session_state.prediction_history:
    history_df = pd.DataFrame(
        st.session_state.prediction_history
    )
    st.dataframe(
        history_df,
        use_container_width=True,
        hide_index=True,
    )
st.subheader("📊 Prediction History Summary")

total_predictions = len(st.session_state.prediction_history)

good_predictions = sum(
    1 for item in st.session_state.prediction_history
    if item["Predicted Risk"] == "Good Credit Risk"
)

bad_predictions = sum(
    1 for item in st.session_state.prediction_history
    if item["Predicted Risk"] == "Bad Credit Risk"
)

col1, col2, col3 = st.columns(3)

col1.metric("Total Predictions", total_predictions)
col2.metric("Lower-Risk Predictions", good_predictions)
col3.metric("Higher-Risk Predictions", bad_predictions)

if st.session_state.prediction_history:
    history_df = pd.DataFrame(
        st.session_state.prediction_history
    )

    st.dataframe(
        history_df,
        use_container_width=True,
        hide_index=True
    )
else:
    st.info(
        "No predictions yet. Enter applicant details "
        "and make your first prediction!"
    )
if st.session_state.prediction_history:
    if st.button("🗑️ Clear Prediction History"):
        st.session_state.prediction_history = []
        st.rerun()
st.divider()
st.caption("Credit Scoring Model | Machine Learning Project")   