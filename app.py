import streamlit as st
import pickle
import pandas as pd

# Load models and encoders
models = pickle.load(open("models.pkl", "rb"))
encoders = pickle.load(open("encoders.pkl", "rb"))

st.title("Salary Prediction App")
st.write("Enter details to predict salary")

# Select model
model_name = st.selectbox("Choose Model", list(models.keys()))
model = models[model_name]

# Input fields
age = st.number_input("Age", min_value=18, max_value=65, value=25)
experience = st.number_input("Years of Experience", min_value=0.0, max_value=40.0, value=1.0)

# --------- FIXED ENCODER HANDLING ---------
keys = list(encoders.keys())

gender_key = None
education_key = None
job_key = None

# Detect correct keys automatically
for key in keys:
    k = key.lower().strip()
    if "gender" in k:
        gender_key = key
    elif "education" in k:
        education_key = key
    elif "job" in k:
        job_key = key

# Safety check
if not (gender_key and education_key and job_key):
    st.error("Encoder keys not found correctly. Please check encoders.pkl")
    st.stop()

# Dropdowns
gender = st.selectbox("Gender", encoders[gender_key].classes_)
education = st.selectbox("Education Level", encoders[education_key].classes_)
job = st.selectbox("Job Title", encoders[job_key].classes_)

# --------- PREDICTION ---------
if st.button("Predict Salary"):
    try:
        # Encode inputs
        gender_enc = encoders[gender_key].transform([gender])[0]
        education_enc = encoders[education_key].transform([education])[0]
        job_enc = encoders[job_key].transform([job])[0]

        # Create input dataframe (must match training order)
        input_data = pd.DataFrame([[age, gender_enc, education_enc, job_enc, experience]],
                                  columns=["Age", "Gender", "Education Level", "Job Title", "Years of Experience"])

        # Predict
        prediction = model.predict(input_data)

        st.success(f"Predicted Salary: {prediction[0]:,.2f}")

    except Exception as e:
        st.error(f"Error: {e}")
