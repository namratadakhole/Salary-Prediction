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

# Inputs
age = st.number_input("Age", min_value=18, max_value=65, value=25)
experience = st.number_input("Years of Experience", min_value=0.0, max_value=40.0, value=1.0)

# --------- FORCE SAFE ENCODER USAGE ---------
encoder_list = list(encoders.values())

# Check at least 3 encoders exist
if len(encoder_list) < 3:
    st.error("Encoders file is incorrect. Recreate encoders.pkl")
    st.stop()

gender_encoder = encoder_list[0]
education_encoder = encoder_list[1]
job_encoder = encoder_list[2]

# Dropdowns
gender = st.selectbox("Gender", gender_encoder.classes_)
education = st.selectbox("Education Level", education_encoder.classes_)
job = st.selectbox("Job Title", job_encoder.classes_)

# Prediction
if st.button("Predict Salary"):
    try:
        gender_enc = gender_encoder.transform([gender])[0]
        education_enc = education_encoder.transform([education])[0]
        job_enc = job_encoder.transform([job])[0]

        input_data = pd.DataFrame([[age, gender_enc, education_enc, job_enc, experience]],
                                  columns=["Age", "Gender", "Education Level", "Job Title", "Years of Experience"])

        prediction = model.predict(input_data)

        st.success(f"Predicted Salary: {prediction[0]:,.2f}")

    except Exception as e:
        st.error(f"Error: {e}")
