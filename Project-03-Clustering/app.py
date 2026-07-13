import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Customer Segmentation Predictor", page_icon="🛍️", layout="wide")

# ---------------------------------------------------------
# Load pre-trained artifact (scaler + kmeans + cluster profile)
# ---------------------------------------------------------
artifact = joblib.load("model.pkl")
scaler = artifact["scaler"]
kmeans = artifact["kmeans"]
features = artifact["features"]
cluster_profile = artifact["cluster_profile"]

# ---------------------------------------------------------
# Map cluster number -> meaningful name, based on YOUR actual results
# ---------------------------------------------------------
cluster_names = {
    3: "💎 Premium Customers (High Income, High Spending)",
    2: "🏦 High Income, Low Engagement",
    4: "🎯 Aspirational Spenders (Low Income, High Spending)",
    5: "💰 Budget-Conscious (Low Income, Low Spending)",
    0: "🧑‍🦳 Older, Moderate Spenders",
    1: "🧑 Younger, Moderate Spenders"
}

# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------
with st.sidebar:
    st.title("ℹ️ About")
    st.write("Predicts a customer's segment using a pre-trained K-Means model (k=6).")

st.title("🛍️ Customer Segmentation Predictor")
st.write("Enter a customer's details to find their segment.")
st.divider()

# ---------------------------------------------------------
# Inputs
# ---------------------------------------------------------
left, right = st.columns(2)
with left:
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    income = st.number_input("Annual Income (k$)", min_value=0, max_value=200, value=60)
with right:
    spending = st.slider("Spending Score (1-100)", 1, 100, 50)

input_df = pd.DataFrame({
    "Age": [age],
    "Annual Income (k$)": [income],
    "Spending Score (1-100)": [spending]
})[features]

st.divider()

# ---------------------------------------------------------
# Predict
# ---------------------------------------------------------
if st.button("🔍 Predict Segment", use_container_width=True):
    input_scaled = scaler.transform(input_df)
    predicted_cluster = kmeans.predict(input_scaled)[0]
    segment_name = cluster_names.get(predicted_cluster, f"Cluster {predicted_cluster}")

    st.header("Prediction Result")
    st.success(f"This customer belongs to: **{segment_name}**")

    st.subheader("Typical Profile for This Segment")
    st.table(cluster_profile.loc[[predicted_cluster]].round(1))

    with st.expander("📄 View Input Data"):
        st.dataframe(input_df, use_container_width=True)