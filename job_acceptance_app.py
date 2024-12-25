import streamlit as st
import pandas as pd
import numpy as np
import joblib
model = joblib.load('hiring.joblib')
scaler = joblib.load('scaler.joblib')
st.set_page_config(page_title = "EquiSelect AI", page_icon = "⚖️")
st.title("EquiSelect AI ⚖️")
st.header("Merit Matters, :rainbow[Diversity] Delivers")
st.subheader("This AI was created to help Employers hire employees without prejudice or any type of discrimination.")
st.write("Pick the employee's  details in the text boxes below, and we'll decide whether they should be accepted or not! ")

education_values = ["High School", "Bachelor's", "Master's"]
awards_values = ["Local", "National", "Regional", "Global"]
role_values = ["Data Science","Java Developer","Network Administrator", "Web Designer","software engineer","Systems Administrator", "IT support specialist"]
education_levels = st.selectbox(" Highest education level?", education_values)
awards_categories = st.selectbox("What is the highest level of award you have gotten?", awards_values)
jobs = st.slider("How many jobs have you had before this?", 1,5,1)
job_role = st.selectbox("What job are you applying for?", role_values)
education_map = {education: idx for idx, education in enumerate(education_values)}
awards_map = {award: idx for idx, award in enumerate(awards_values)}
role_map = {job: idx for idx, job in enumerate(role_values)}
education_num = education_map[education_levels]
awards_num = awards_map[awards_categories]
role_num = role_map[job_role]
st.write(f"Education Map: {education_map}")
st.write(f"Awards Map: {awards_map}")

# Create a DataFrame for the input data
input_data = pd.DataFrame({
    'Education': [education_num],
    'Awards': [awards_num],
    'Previous Jobs': [jobs]
})

# Standardize the input data
input_data_scaled = scaler.transform(input_data)
# Predict acceptance/rejection
if st.button('Predict'):
    prediction = model.predict(input_data_scaled)[0]
    st.write(f"predict result: {prediction}") 
    if prediction == 'Accepted':
        st.success('Accepted')
    else:
        st.error('Rejected')
