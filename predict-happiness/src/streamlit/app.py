# app.py
import streamlit as st
import pandas as pd
import catboost
from catboost import CatBoostRegressor
# import joblib
import pickle as pk
import json
# Load the trained model
#model = joblib.load('../../../cbr_model.pkl')
with open("../../../cbr_model.pkl", "rb") as file:
    model = pk.load(file)
# Title
st.title('ML Model Deployment with Streamlit')
test = pd.read_csv("../../../test_data_2019.csv")



# Read dictionary from a file
with open('../../../country_labels_map.txt', 'r') as file:
    country_labels_map = json.load(file)


#test = test.drop(index = test[test['Country'].isnull()==True].index.tolist(),axis = 0)
# User inputs
st.subheader('Input Features')
feature1 = st.text_input('Enter country')#, min_value=0, max_value=10, value=5)
#print(feature1)
#st.write(type(feature1))
#feature2 = st.number_input('Feature 2', min_value=0, max_value=100, value=50)
feature2 = "Benin"
#st.write(f'You entered: {feature1}')
#feature1 = str(feature1)
#country_label = country_labels_map[f'{feature1}']
# if feature1 is not None:
#     st.write(f'You entered: {feature1}')
    
# Predict button
if st.button('Predict'):
    country_label = country_labels_map[feature1]
   # input_data = pd.DataFrame({'feature1': [feature1], 'feature2': [feature2]})
    prediction = model.predict(test)
    st.write(f'Prediction for Happiness Score: {prediction[country_label]}')

# Main section
st.write("This is a simple machine learning model deployment using Streamlit.")
