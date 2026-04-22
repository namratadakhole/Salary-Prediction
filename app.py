import streamlit as st
import pickle
import pandas as pd

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="Salary Prediction System",
    layout="centered"
)

# ---------- LOAD FILES ----------
models = pickle.load(open("models.pkl", "rb"))
encoders = pickle.load(open("encoders.pkl", "rb"))

# ---------- CUSTOM CSS ----------
st.markdown("""
<style>
body {
    background: linear-gradient(120deg, #eef2ff, #f8fafc);
}

.block-container {
    padding-top: 2rem;
}

.header {
    font-size: 34px;
    font-weight: 600;
    text-align: center;
    color: #1e3a8a;
    margin-bottom: 5px;
}

.subtext {
    text-align: center;
    color: #475569;
    margin-bottom: 25px;
}

.section {
    background: white;
    padding: 25px;
    border-radius: 12px;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.06);
    margin-bottom: 20px;
}

.section-title {
    font-size: 18px;
    font-weight: 600;
    color: #1e3a8a;
    margin-bottom: 10px;
}

.result {
    background: linear-gradient(120deg, #2563eb, #1e40af);
    padding: 25px;
    border-radius: 12px;
    text-align: center;
    color: white;
    margin-top: 15px;
}

.stButton>button {
    width: 100%;
    height: 45px;
    border-radius: 8px;
    background-color: #2563eb;
    color: white;
    font-weight: 500;
    border: none;
}
</style>
""", unsafe_allow_html=True)

# ---------- HEADER ----------
st.markdown("<div class='header'>Salary Prediction System</div>", unsafe_allow_html=True)
st.markdown("<div class='subtext'>Predict salary using machine learning models</div>", unsafe_allow_html=True)

# ---------- MODEL SELECTION ----------
with st.container():
    st.markdown("<div class='section'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Model Selection</div>", unsafe_allow_html=True)

    model_name = st.selectbox("Select Model", list(models.keys()))
    model = models[model_name]

    st.markdown("</div>", unsafe_allow_html=True)

# ---------- INPUT FORM ----------
with st.form("prediction_form"):
    st.markdown("<div class='section'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Enter Details</div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", 18, 65, 25)
        gender = st.selectbox("Gender", encoders["Gender"].classes_)

    with col2:
        experience = st.number_input("Years of Experience", 0.0, 40.0, 1.0)
        education = st.selectbox("Education Level", encoders["Education Level"].classes_)

    job = st.selectbox("Job Title", encoders["Job Title"].classes_)

    submitted = st.form_submit_button("Predict Salary")

    st.markdown("</div>", unsafe_allow_html=True)

# ---------- PREDICTION ----------
if submitted:
    try:
        gender_enc = encoders["Gender"].transform([gender])[0]
        education_enc = encoders["Education Level"].transform([education])[0]
        job_enc = encoders["Job Title"].transform([job])[0]

        input_data = pd.DataFrame([[age, gender_enc, education_enc, job_enc, experience]],
                                  columns=["Age", "Gender", "Education Level", "Job Title", "Years of Experience"])

        prediction = model.predict(input_data)[0]

        st.markdown("<div class='result'>", unsafe_allow_html=True)
        st.markdown("<h3>Predicted Salary</h3>", unsafe_allow_html=True)
        st.markdown(f"<h1>{prediction:,.2f}</h1>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error: {e}")

# ---------- FOOTER ----------
st.markdown(
    "<p style='text-align:center; color:#64748b; margin-top:40px;'>Machine Learning Application using Streamlit</p>",
    unsafe_allow_html=True
)
