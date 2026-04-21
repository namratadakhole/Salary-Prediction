import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder

# --- Load dataset and fit encoders ---
@st.cache_resource
def load_data_and_encoders():
    df = pd.read_csv('Salary_Data.csv')  # make sure this file is in GitHub

    # Handle missing values
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col].fillna(df[col].mode()[0], inplace=True)
        else:
            df[col].fillna(df[col].mean(), inplace=True)

    df.drop_duplicates(inplace=True)

    # Fit encoders
    gender_encoder = LabelEncoder()
    edu_encoder = LabelEncoder()
    job_encoder = LabelEncoder()

    gender_encoder.fit(df['Gender'])
    edu_encoder.fit(df['Education Level'])
    job_encoder.fit(df['Job Title'])

    return gender_encoder, edu_encoder, job_encoder

# --- Load model ---
@st.cache_resource
def load_model():
    return pickle.load(open('best_model.pkl', 'rb'))

model = load_model()
gender_encoder, edu_encoder, job_encoder = load_data_and_encoders()

# --- UI ---
st.title("💼 Salary Prediction App")

age = st.slider("Age", 18, 65, 30)
gender = st.selectbox("Gender", gender_encoder.classes_)
education = st.selectbox("Education Level", edu_encoder.classes_)
job = st.selectbox("Job Title", job_encoder.classes_)
experience = st.slider("Years of Experience", 0.0, 40.0, 5.0, 0.5)

# --- Prediction ---
if st.button("Predict Salary"):
    try:
        gender_enc = gender_encoder.transform([gender])[0]
        edu_enc = edu_encoder.transform([education])[0]
        job_enc = job_encoder.transform([job])[0]

        input_df = pd.DataFrame(
            [[age, gender_enc, edu_enc, job_enc, experience]],
            columns=['Age', 'Gender', 'Education Level', 'Job Title', 'Years of Experience']
        )

        prediction = model.predict(input_df)[0]
        st.success(f"💰 Predicted Salary: ${prediction:,.2f}")

    except Exception as e:
        st.error(f"Error: {e}")
