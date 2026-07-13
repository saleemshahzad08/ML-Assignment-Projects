

import streamlit as st              # For UI of Web Application
import pandas as pd                 # For Creating a Data Frame
import joblib                       # For Loading the Trained Model

# -------------------------------------------------------
# Loading Model
# -------------------------------------------------------

model = joblib.load('model.pkl')

st.title('MedCost Predict')

st.write("Predict your health insurance charges based on age, BMI, smoking habits & more!")

age = st.slider(
    "Age",
    min_value = 0,
    max_value = 100,
    value = 25
)

sex = st.selectbox(
    "Workclass",
    ['female', 'male']
)

bmi = st.slider(
    "BMI",
    min_value = 15,
    max_value = 55,
    value = 25
)

children = st.slider(
    "No. of children",
    min_value = 0,
    max_value = 10,
    value = 5
)

smoker = st.selectbox(
    "Smoker",
    ['yes', 'no']
)

region = st.selectbox(
    "Region",
    ['southwest', 'southeast', 'northwest', 'northeast']
)

input_data = pd.DataFrame({

    "age": [age],
    "sex": [sex],
    "bmi": [bmi],
    "children": [children],
    "smoker": [smoker],
    "region": [region]
})

st.subheader("Input Sent to the Model")

st.dataframe(input_data)

if st.button("Predict"):

    # Predict survival.
    prediction = model.predict(input_data)[0]

    st.subheader("Predicted Health Insurance Charges")

    st.write(prediction)
