import streamlit as st
import pandas as pd
import pickle
import numpy as np

# --- Load Model and Encoders ---
@st.cache_resource
def load_artifacts():
    model = pickle.load(open('best_model.pkl', 'rb'))
    gender_encoder = pickle.load(open('gender_encoder.pkl', 'rb'))
    edu_encoder = pickle.load(open('edu_encoder.pkl', 'rb'))
    job_encoder = pickle.load(open('job_encoder.pkl', 'rb'))
    return model, gender_encoder, edu_encoder, job_encoder

model, gender_encoder, edu_encoder, job_encoder = load_artifacts()

# --- App UI ---
st.set_page_config(page_title="Salary Predictor", page_icon="💼")

st.title("💼 Salary Prediction App")
st.write("Enter the details below to predict salary.")

# Inputs
age = st.slider("Age", 18, 65, 30)

gender = st.selectbox("Gender", gender_encoder.classes_)
education = st.selectbox("Education Level", edu_encoder.classes_)
job = st.selectbox("Job Title", job_encoder.classes_)

experience = st.slider("Years of Experience", 0.0, 40.0, 5.0, 0.5)

# Prediction
if st.button("Predict Salary"):
    try:
        gender_enc = gender_encoder.transform([gender])[0]
        edu_enc = edu_encoder.transform([education])[0]
        job_enc = job_encoder.transform([job])[0]

        input_data = pd.DataFrame(
            [[age, gender_enc, edu_enc, job_enc, experience]],
            columns=['Age', 'Gender', 'Education Level', 'Job Title', 'Years of Experience']
        )

        prediction = model.predict(input_data)[0]

        st.success(f"💰 Predicted Salary: ${prediction:,.2f}")

    except Exception as e:
        st.error(f"Error: {e}")
