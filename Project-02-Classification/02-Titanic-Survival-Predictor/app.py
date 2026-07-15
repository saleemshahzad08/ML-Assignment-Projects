# ==========================================================
# Titanic Survival Prediction App
#
# This application loads a trained Scikit-learn Pipeline
# and predicts whether a passenger would survive.
#
# The Pipeline already performs:
#   • Missing value imputation
#   • One-Hot Encoding
#   • Feature Scaling
#   • Logistic Regression Prediction
#
# Therefore, we only provide the ORIGINAL features.
# ==========================================================

# Import Streamlit for building the web application.
import streamlit as st

# Import Pandas to create a DataFrame.
import pandas as pd

# Joblib is used to load the trained model.
import joblib

# ----------------------------------------------------------
# Load the trained machine learning model.
# ----------------------------------------------------------

model = joblib.load("Project-02-Classification/02-Titanic-Survival-Predictor/model.pkl")

# ----------------------------------------------------------
# App Title
# ----------------------------------------------------------

st.title("🚢 Titanic Survival Prediction")

st.write(
    "Enter the passenger details below and click **Predict** "
    "to estimate whether the passenger would survive."
)

# ----------------------------------------------------------
# User Inputs
# ----------------------------------------------------------

sex = st.selectbox(
    "Sex",
    ["male", "female"]
)

embarked = st.selectbox(
    "Embarked Port",
    ["C", "Q", "S"]
)

pclass = st.slider(
    "Passenger Class",
    min_value=1,
    max_value=3,
    value=3
)

age = st.number_input(
    "Age",
    min_value=0,
    max_value=100,
    value=25
)

fare = st.number_input(
    "Fare",
    min_value=0.0,
    value=32.0
)

sibsp = st.slider(
    "Number of Siblings / Spouses",
    min_value=0,
    max_value=10,
    value=0
)

parch = st.slider(
    "Number of Parents / Children",
    min_value=0,
    max_value=10,
    value=0
)

alone = st.selectbox(
    "Travelling Alone?",
    ["Yes", "No"]
)

# ----------------------------------------------------------
# Convert "Yes" / "No" into integers.
#
# During training, the "alone" column was converted into:
#
# True  -> 1
# False -> 0
# ----------------------------------------------------------

alone = 1 if alone == "Yes" else 0

# ----------------------------------------------------------
# Create a DataFrame.
#
# The column names must exactly match those used during
# model training.
# ----------------------------------------------------------

input_data = pd.DataFrame({

    "pclass": [pclass],
    "sex": [sex],
    "age": [age],
    "fare": [fare],
    "sibsp": [sibsp],
    "parch": [parch],
    "embarked": [embarked],
    "alone": [alone]

})

st.subheader("Input Sent to the Pipeline")

st.dataframe(input_data)

# ----------------------------------------------------------
# Predict Button
# ----------------------------------------------------------

if st.button("Predict"):

    # Predict survival.
    prediction = model.predict(input_data)[0]

    # Predict probability.
    probability = model.predict_proba(input_data)[0]

    st.subheader("Prediction Result")

    if prediction == 1:

        st.success("🎉 The passenger is predicted to SURVIVE.")

        st.write(
            f"**Probability of Survival:** "
            f"{probability[1]:.2%}"
        )

    else:

        st.error("❌ The passenger is predicted NOT to survive.")

        st.write(
            f"**Probability of Survival:** "
            f"{probability[0]:.2%}"
        )
