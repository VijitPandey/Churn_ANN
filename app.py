import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pickle
import tensorflow as tf

# Load the model
model = tf.keras.models.load_model('model.h5')

# Load the scaler
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Load the encoder
with open('one_hot_encoder.pkl', 'rb') as f:
    ohe = pickle.load(f)

with open('label_encoder.pkl', 'rb') as f:
    l_encoder = pickle.load(f)

st.title('Customer Churn Prediction')
st.write('This app predicts whether a customer will churn or not based on their data.')

st.write('Please enter the customer data below:')

# Create input fields for customer data
geography = st.selectbox('Geography', ohe.categories_[0])
gender = st.selectbox('Gender', l_encoder.classes_)
age = st.slider('Age', min_value=18, max_value=100, value=30)
tenure = st.slider('Tenure', min_value=0, max_value=10, value=5)
balance = st.number_input('Balance')
num_of_products = st.slider('Number of Products', min_value=1, max_value=4, value=2)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])
estimated_salary = st.number_input('Estimated Salary')
credit_score = st.number_input('Credit Score')


input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [l_encoder.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

# geography_encoded = ohe.transform([[geography]]).toarray()
# input_data = pd.concat([input_data, pd.DataFrame(geography_encoded, columns=ohe.get_feature_names_out(['Geography']))], axis=1)
# One-hot encode 'Geography'
geo_encoded = ohe.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=ohe.get_feature_names_out(['Geography']))

# Combine one-hot encoded columns with input data
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)
# Scale the data
input_data_scaled = scaler.transform(input_data)

# Make prediction
prediction = model.predict(input_data_scaled)
prediction_proba = prediction[0][0]


# if st.button('Predict'):
if prediction_proba > 0.5:
    st.write(f'The customer is likely to churn with a probability of {prediction_proba:.2f}')
else:
    st.write(f'The customer is unlikely to churn with a probability of {1 - prediction_proba:.2f}')