import streamlit as st
import pickle
import pandas as pd

# Page config
st.set_page_config(page_title="Salary Predictor", layout="wide")

# Load model & encoders
models = pickle.load(open("models.pkl", "rb"))
encoders = pickle.load(open("encoders.pkl", "rb"))

# ---------- CUSTOM CSS ----------
st.markdown("""
<style>
body {
    background-color: #0f172a;
    color: white;
}
.main {
    background-color: #0f172a;
}

.title {
    text-align: center;
    font-size: 40px;
    font-weight: 600;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    font-size: 16px;
    color: #94a3b8;
    margin-bottom: 30px;
}

.card {
    background: #1e293b;
    padding: 20px;
    border-radius: 12px;
    margin-bottom: 20px;
}

.stButton>button {
    background-color: #2563eb;
    color: white;
    border-radius: 8px;
    height: 3em;
    width: 100%;
    font-size: 16px;
}

.result-box {
    text-align: center;
    padding: 20px;
    background: #1e293b;
    border-radius: 12px;
    margin-top: 20px;
}
</style>
""", unsafe_allow_html=True)

# ---------- HEADER ----------
st.markdown("<div class='title'>Salary Prediction</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Predict salary based on professional details</div>", unsafe_allow_html=True)

# ---------- SIDEBAR ----------
st.sidebar.title("Model Selection")
model_name = st.sidebar.selectbox("Choose Model", list(models.keys()))
model = models[model_name]

st.sidebar.write("Selected Model:")
st.sidebar.write(model_name)

# ---------- INPUT SECTION ----------
col1, col2 = st.columns(2)

with col1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    age = st.slider("Age", 18, 65, 25)
    gender = st.selectbox("Gender", encoders["Gender"].classes_)
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    experience = st.slider("Years of Experience", 0.0, 40.0, 1.0)
    education = st.selectbox("Education Level", encoders["Education Level"].classes_)
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div class='card'>", unsafe_allow_html=True)
job = st.selectbox("Job Title", encoders["Job Title"].classes_)
st.markdown("</div>", unsafe_allow_html=True)

# ---------- PREDICTION ----------
if st.button("Predict Salary"):
    try:
        gender_enc = encoders["Gender"].transform([gender])[0]
        education_enc = encoders["Education Level"].transform([education])[0]
        job_enc = encoders["Job Title"].transform([job])[0]

        input_data = pd.DataFrame([[age, gender_enc, education_enc, job_enc, experience]],
                                  columns=["Age", "Gender", "Education Level", "Job Title", "Years of Experience"])

        prediction = model.predict(input_data)[0]

        st.markdown("<div class='result-box'>", unsafe_allow_html=True)
        st.markdown(f"<h2>Predicted Salary</h2>", unsafe_allow_html=True)
        st.markdown(f"<h1>{prediction:,.2f}</h1>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error: {e}")

# ---------- FOOTER ----------
st.markdown("<p style='text-align:center; color:#94a3b8; margin-top:40px;'>Machine Learning Application using Streamlit</p>", unsafe_allow_html=True)
