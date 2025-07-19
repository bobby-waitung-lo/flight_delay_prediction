import streamlit as st
import pandas as pd
import joblib

# Load pipeline
model = joblib.load("delay_pipeline.pkl")

# Page title
st.title("✈️ Flight Delay Prediction")

# Input form
with st.form("predict_form"):
    origin = st.text_input("US Origin Airport Code (e.g., JFK)")
    dest = st.text_input("US Destination Airport Code (e.g., LAX)")
    dep_delay = st.number_input("Departure Delay (minutes)", value=0)

    submit = st.form_submit_button("Predict")

# Prepare input and predict
if submit:
    input_df = pd.DataFrame({
        "ORIGIN": [origin],
        "DEST": [dest],
        "DEP_DELAY": [dep_delay]
    })

    prediction = model.predict_proba(input_df)[0][1]
    st.metric("Predicted Probability of Delay", f"{prediction:.2%}")