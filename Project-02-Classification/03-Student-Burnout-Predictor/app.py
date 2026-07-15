import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

# ---------------------------------------------------------
# 1. Configure the Streamlit page.
# This MUST be the first Streamlit command.
# ---------------------------------------------------------
st.set_page_config(
    page_title="Student Burnout Level Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# 2. Load the trained Scikit-learn pipeline.
# The pipeline already contains preprocessing + model.
# ---------------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load("Project-02-Classification/03-Student-Burnout-Predictor/model.pkl")

model = load_model()

# ---------------------------------------------------------
# 3. Sidebar
# ---------------------------------------------------------
with st.sidebar:
    st.title("ℹ️ About")
    st.write("""
This app predicts a student's **Burnout Level**
(*Low / Medium / High*) using academic, lifestyle and
psychological indicators.

**Dataset:** Student Burnout & Dropout Dataset

**Target:** Burnout Level (Multiclass)

**Model:** Best estimator selected via GridSearchCV
(Logistic Regression, Decision Tree, Random Forest, SVC)

**Deployment:** Streamlit
""")
    st.divider()
    st.write("Developed as part of an AI Engineering course.")

# ---------------------------------------------------------
# 4. Heading
# ---------------------------------------------------------
st.title("🎓 Student Burnout Level Predictor")

st.write("""
Fill in a student's academic, lifestyle and psychological
details below to predict their **burnout risk level**
using a Machine Learning model.
""")

st.divider()

# ---------------------------------------------------------
# 5. Input sections
# ---------------------------------------------------------

st.subheader("🧑‍🎓 Personal & Academic Info")
c1, c2, c3 = st.columns(3)

with c1:
    age = st.number_input(
        "Age",
        min_value=17, max_value=25, value=20,
        help="Student's age in years."
    )
    gender = st.selectbox(
        "Gender",
        ["Female", "Male", "Other"],
        help="Student's gender."
    )

with c2:
    year_of_study = st.slider(
        "Year of Study",
        1, 4, 2,
        help="Current academic year."
    )
    department = st.selectbox(
        "Department",
        ["Arts", "Business", "Engineering", "Law", "Medicine", "Science"],
        help="Academic department."
    )

with c3:
    previous_gpa = st.number_input(
        "Previous GPA",
        min_value=0.0, max_value=10.0, value=7.5, step=0.01,
        help="GPA from the previous term (0-10 scale)."
    )
    backlogs = st.number_input(
        "Backlogs",
        min_value=0, max_value=6, value=0,
        help="Number of pending backlogs/failed courses."
    )

st.divider()

st.subheader("🌙 Lifestyle & Habits")
c4, c5, c6 = st.columns(3)

with c4:
    attendance_percent = st.slider(
        "Attendance (%)",
        30.0, 100.0, 80.0,
        help="Overall class attendance percentage."
    )
    study_hours = st.slider(
        "Study Hours / Day",
        0.0, 8.1, 3.0, step=0.1,
        help="Average hours spent studying per day."
    )

with c5:
    sleep_hours = st.slider(
        "Sleep Hours / Night",
        2.0, 10.0, 7.0, step=0.1,
        help="Average hours of sleep per night."
    )
    screen_time = st.slider(
        "Screen Time (hrs/day)",
        0.5, 12.6, 5.0, step=0.1,
        help="Average daily screen time."
    )

with c6:
    exercise_freq = st.slider(
        "Exercise Frequency (per week)",
        0, 7, 2,
        help="Number of days exercised per week."
    )
    residence_type = st.selectbox(
        "Residence Type",
        ["Day Scholar", "Hostel", "PG/Rented"],
        help="Where the student currently resides."
    )

st.divider()

st.subheader("🧠 Psychological & Support Factors")
c7, c8, c9 = st.columns(3)

with c7:
    stress_level = st.slider(
        "Stress Level",
        0.1, 10.0, 5.0, step=0.1,
        help="Self-reported overall stress level (0-10)."
    )
    anxiety_score = st.slider(
        "Anxiety Score",
        0.0, 10.0, 5.0, step=0.1,
        help="Self-reported anxiety score (0-10)."
    )
    motivation_score = st.slider(
        "Motivation Score",
        0.0, 10.0, 5.0, step=0.1,
        help="Self-reported motivation level (0-10)."
    )

with c8:
    peer_pressure = st.slider(
        "Peer Pressure Score",
        0.0, 9.8, 5.0, step=0.1,
        help="Perceived peer pressure (0-10)."
    )
    social_activity = st.slider(
        "Social Activity Score",
        0.0, 10.0, 5.0, step=0.1,
        help="Level of participation in social activities (0-10)."
    )
    family_support = st.slider(
        "Family Support Score",
        0.0, 10.0, 5.0, step=0.1,
        help="Perceived family support level (0-10)."
    )

with c9:
    financial_stress = st.slider(
        "Financial Stress Score",
        0.0, 10.0, 5.0, step=0.1,
        help="Self-reported financial stress (0-10)."
    )
    family_income = st.selectbox(
        "Family Income Bracket",
        ["Low", "Lower-Middle", "Middle", "Upper-Middle", "High"],
        index=2,
        help="Household income bracket."
    )
    part_time_job = st.selectbox(
        "Part-Time Job?",
        ["No", "Yes"]
    )
    counseling_access = st.selectbox(
        "Access to Counseling?",
        ["No", "Yes"]
    )

# ---------------------------------------------------------
# 6. Create DataFrame
# IMPORTANT:
# Column names must match training data.
# ---------------------------------------------------------
input_df = pd.DataFrame({
    "Age": [age],
    "Gender": [gender],
    "Year_of_Study": [year_of_study],
    "Department": [department],
    "Residence_Type": [residence_type],
    "Attendance_Percent": [attendance_percent],
    "Study_Hours_Per_Day": [study_hours],
    "Previous_GPA": [previous_gpa],
    "Backlogs": [backlogs],
    "Sleep_Hours": [sleep_hours],
    "Screen_Time_Hours": [screen_time],
    "Exercise_Freq_Per_Week": [exercise_freq],
    "Social_Activity_Score": [social_activity],
    "Part_Time_Job": [part_time_job],
    "Family_Income_Bracket": [family_income],
    "Financial_Stress_Score": [financial_stress],
    "Family_Support_Score": [family_support],
    "Stress_Level": [stress_level],
    "Anxiety_Score": [anxiety_score],
    "Motivation_Score": [motivation_score],
    "Peer_Pressure_Score": [peer_pressure],
    "Counseling_Access": [counseling_access]
})

st.divider()

# ---------------------------------------------------------
# 7. Prediction Button
# ---------------------------------------------------------
if st.button("🔍 Predict Burnout Level", use_container_width=True):

    prediction = model.predict(input_df)[0]

    st.header("Prediction Result")

    result_style = {
        "Low": ("success", "🟢", "Low burnout risk. Keep up the healthy balance!"),
        "Medium": ("warning", "🟡", "Moderate burnout risk. Some lifestyle adjustments could help."),
        "High": ("error", "🔴", "High burnout risk. Consider seeking support and reducing load.")
    }
    kind, emoji, message = result_style.get(prediction, ("info", "ℹ️", ""))
    getattr(st, kind)(f"{emoji} Predicted Burnout Level: **{prediction}** — {message}")

    # Relative confidence from the model's decision function (SVC has no
    # predict_proba here since probability=False was used during training).
    if hasattr(model, "decision_function"):
        try:
            scores = model.decision_function(input_df)[0]
            classes = model.classes_
            exp_scores = np.exp(scores - np.max(scores))
            rel_confidence = exp_scores / exp_scores.sum()

            st.subheader("Relative Confidence")
            st.caption(
                "Derived from the model's decision scores (softmax-normalized). "
                "Not a calibrated probability."
            )
            conf_df = pd.DataFrame({
                "Burnout Level": classes,
                "Relative Confidence": rel_confidence
            }).sort_values("Relative Confidence", ascending=False)

            for _, row in conf_df.iterrows():
                st.write(f"**{row['Burnout Level']}**")
                st.progress(float(row["Relative Confidence"]))
        except Exception:
            pass

    st.subheader("Prediction Summary")

    summary = pd.DataFrame({
        "Feature": [
            "Age", "Gender", "Year of Study", "Department", "Residence Type",
            "Attendance (%)", "Study Hours/Day", "Previous GPA", "Backlogs",
            "Sleep Hours", "Screen Time (hrs)", "Exercise Freq/Week",
            "Social Activity Score", "Part-Time Job", "Family Income Bracket",
            "Financial Stress Score", "Family Support Score", "Stress Level",
            "Anxiety Score", "Motivation Score", "Peer Pressure Score",
            "Counseling Access"
        ],
        "Value": [
            age, gender, year_of_study, department, residence_type,
            attendance_percent, study_hours, previous_gpa, backlogs,
            sleep_hours, screen_time, exercise_freq,
            social_activity, part_time_job, family_income,
            financial_stress, family_support, stress_level,
            anxiety_score, motivation_score, peer_pressure,
            counseling_access
        ]
    })

    st.table(summary)

    with st.expander("📄 View Input Data"):
        st.dataframe(input_df, use_container_width=True)

    with st.expander("🤖 Model Information"):
        st.write("""
**Algorithm:** Best estimator selected via GridSearchCV
(compared Logistic Regression, Decision Tree, Random Forest, SVC)

**Selected Model:** Support Vector Classifier (linear kernel)

**Evaluation on held-out test set (20% split):**

| Metric | Score |
|---|---|
| Accuracy | 70.00% |
| F1 Score (weighted) | 70.12% |

**Per-class performance:**

| Class | Precision | Recall | F1-score |
|---|---|---|---|
| High | 0.75 | 0.60 | 0.67 |
| Low | 0.80 | 0.81 | 0.80 |
| Medium | 0.55 | 0.62 | 0.58 |
""")

    st.caption(
        f"Prediction generated on "
        f"{datetime.now().strftime('%d %B %Y, %I:%M:%S %p')}"
    )

st.divider()
st.caption("Built with ❤️ using Streamlit and Scikit-learn.")
