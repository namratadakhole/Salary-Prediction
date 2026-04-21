import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
import numpy as np

# --- Helper function to load and preprocess data (for fitting LabelEncoders) ---
@st.cache_resource
def load_and_preprocess_for_encoders():
    # Load the original dataset
    original_df = pd.read_csv('/content/Salary_Data (1).csv')

    # Impute missing values (same logic as in notebook)
    for column in original_df.columns:
        if original_df[column].dtype == 'object':
            mode_val = original_df[column].mode()[0]
            original_df[column] = original_df[column].fillna(mode_val)
        else:
            mean_val = original_df[column].mean()
            original_df[column] = original_df[column].fillna(mean_val)
    
    # Drop duplicates (important for consistent data used for fitting encoders)
    original_df.drop_duplicates(inplace=True)

    # Fit LabelEncoders for each categorical column
    gender_encoder = LabelEncoder()
    edu_level_encoder = LabelEncoder()
    job_title_encoder = LabelEncoder()

    gender_encoder.fit(original_df['Gender'])
    edu_level_encoder.fit(original_df['Education Level'])
    job_title_encoder.fit(original_df['Job Title'])

    return gender_encoder, edu_level_encoder, job_title_encoder

# --- Load Model and Encoders ---
@st.cache_resource
def load_model():
    with open('best_model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

model = load_model()
gender_encoder, edu_level_encoder, job_title_encoder = load_and_preprocess_for_encoders()

# --- Streamlit App ---
st.title("Salary Prediction App")
st.write("Enter the details to predict the salary.")

# Input fields for prediction
age = st.slider("Age", 18, 65, 30)
gender_options = gender_encoder.classes_
gender_input = st.selectbox("Gender", gender_options)

edu_level_options = edu_level_encoder.classes_
edu_level_input = st.selectbox("Education Level", edu_level_options)

job_title_options = job_title_encoder.classes_
job_title_input = st.selectbox("Job Title", job_title_options)

years_experience = st.slider("Years of Experience", 0.0, 40.0, 5.0, 0.5)

# Prediction button
if st.button("Predict Salary"):
    try:
        # Encode categorical features
        gender_encoded = gender_encoder.transform([gender_input])[0]
        edu_level_encoded = edu_level_encoder.transform([edu_level_input])[0]
        job_title_encoded = job_title_encoder.transform([job_title_input])[0]

        # Create input DataFrame (matching the order of X_train)
        # The order of columns in X was: 'Age', 'Gender', 'Education Level', 'Job Title', 'Years of Experience'
        input_data = pd.DataFrame([[age, gender_encoded, edu_level_encoded, job_title_encoded, years_experience]],
                                  columns=['Age', 'Gender', 'Education Level', 'Job Title', 'Years of Experience'])

        prediction = model.predict(input_data)[0]
        st.success(f"Predicted Salary: ${prediction:,.2f}")
    except ValueError as e:
        st.error(f"Error during prediction: {e}. Please ensure all inputs are valid.")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
