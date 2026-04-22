import streamlit as st
import pickle
import pandas as pd

# Load models and encoders
models = pickle.load(open("models.pkl", "rb"))
encoders = pickle.load(open("encoders.pkl", "rb"))
st.write(encoders.keys())

st.title("Salary Prediction App")

st.write("Enter details to predict salary")

# Select model
model_name = st.selectbox("Choose Model", list(models.keys()))
model = models[model_name]

# Input fields
age = st.number_input("Age", min_value=18, max_value=65, value=25)
experience = st.number_input("Years of Experience", min_value=0.0, max_value=40.0, value=1.0)

# Get actual keys from encoders
keys = list(encoders.keys())

# Assign dynamically
gender_key = keys[0]
education_key = keys[1]
job_key = keys[2]

gender = st.selectbox("Gender", encoders[gender_key].classes_)
education = st.selectbox("Education Level", encoders[education_key].classes_)
job = st.selectbox("Job Title", encoders[job_key].classes_)

# Prediction
if st.button("Predict Salary"):
    try:
        # Encode inputs
        gender_enc = encoders["Gender"].transform([gender])[0]
        education_enc = encoders["Education Level"].transform([education])[0]
        job_enc = encoders["Job Title"].transform([job])[0]

        # Create input dataframe (order must match training)
        input_data = pd.DataFrame([[age, gender_enc, education_enc, job_enc, experience]],
                                  columns=["Age", "Gender", "Education Level", "Job Title", "Years of Experience"])

        # Predict
        prediction = model.predict(input_data)

        st.success(f"Predicted Salary: {prediction[0]:,.2f}")

    except Exception as e:
        st.error(f"Error: {e}") 
