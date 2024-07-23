# app.py
import streamlit as st
import pandas as pd
# import joblib
# Load the trained model
# model = joblib.load('model.pkl')

# Title
st.title('ML Model Deployment with Streamlit')

# User inputs
st.subheader('Input Features')
feature1 = st.number_input('Feature 1', min_value=0, max_value=10, value=5)
feature2 = st.number_input('Feature 2', min_value=0, max_value=100, value=50)

# Predict button
if st.button('Predict'):
    input_data = pd.DataFrame({'feature1': [feature1], 'feature2': [feature2]})
    #prediction = model.predict(input_data)
    #st.write(f'Prediction: {prediction[0]}')

# Main section
st.write("This is a simple machine learning model deployment using Streamlit.")
