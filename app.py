import streamlit as st
import pickle
import pandas as pd

# Page config
st.set_page_config(page_title="Salary Predictor", layout="centered")

# Load files
models = pickle.load(open("models.pkl", "rb"))
encoders = pickle.load(open("encoders.pkl", "rb"))

# Header
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>Salary Prediction App</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Predict salary based on your profile</p>", unsafe_allow_html=True)

# Sidebar
st.sidebar.header("⚙️ Settings")
model_name = st.sidebar.selectbox("Choose Model", list(models.keys()))
model = models[model_name]

st.sidebar.write(f"Using: **{model_name}**")

st.write("---")

# Layout using columns
col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 18, 65, 25)
    gender = st.selectbox("Gender", encoders["Gender"].classes_)

with col2:
    experience = st.slider("Years of Experience", 0.0, 40.0, 1.0)
    education = st.selectbox("Education Level", encoders["Education Level"].classes_)

# Full width
job = st.selectbox("Job Title", encoders["Job Title"].classes_)

st.write("")

# Predict button (centered)
predict_btn = st.button("Predict Salary")

if predict_btn:
    try:
        # Encode inputs
        gender_enc = encoders["Gender"].transform([gender])[0]
        education_enc = encoders["Education Level"].transform([education])[0]
        job_enc = encoders["Job Title"].transform([job])[0]

        # Input dataframe
        input_data = pd.DataFrame([[age, gender_enc, education_enc, job_enc, experience]],
                                  columns=["Age", "Gender", "Education Level", "Job Title", "Years of Experience"])

        prediction = model.predict(input_data)[0]

        st.markdown("Predicted Salary")
        st.success(f"₹ {prediction:,.2f}")

    except Exception as e:
        st.error(f"Error: {e}")

st.write("---")

# Footer
st.markdown(
    "<p style='text-align: center; font-size: 12px;'>Built using Streamlit | ML Project</p>",
    unsafe_allow_html=True
)
