import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder

import os
# Disables all GPUs, forcing operations onto the CPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from keras.models import load_model
import streamlit as st


import pickle
# Load the pre-fitted encoder
try:
    with open('oh_encoder.pkl', 'rb') as file:
        oh_encoder = pickle.load(file)
    with open('lbl_encoder.pkl', 'rb') as file:
        lbl_encoder = pickle.load(file)
    with open('scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)
except FileNotFoundError:
    st.error("Error: one of the .pkl files could not be found. Make sure the fitted encoder is in the correct path.")
    st.stop() # Stop the app if the encoder can't be loaded


# # Import model
# model = load_model('./ann.keras')

from pathlib import Path
from keras.src.saving import saving_api
# 1. Define the base directory as the directory containing app.py
BASE_DIR = Path(__file__).resolve().parent

# 2. Construct the absolute path to the model file
# The Path object automatically handles correct path separators ('/')
MODEL_PATH = BASE_DIR / 'ann.keras' 

# 3. Load the model using the absolute path
# IMPORTANT: Use try/except to catch and display the REAL error message, 
# which is often suppressed in the initial Streamlit error page.
try:
    model = saving_api.load_model(
        MODEL_PATH,
        # *** REMINDER: If you have custom functions (like an F1 metric), 
        # you MUST define them and pass them here:
        # custom_objects={'f1_score': your_f1_function}
        custom_objects=None 
    )
    st.success("Model loaded successfully!")
except ValueError as e:
    st.error(f"FATAL Model Loading Error: {e}")
    # Display the full path attempted
    st.code(f"Attempted to load from path: {MODEL_PATH}")
    st.stop() # Stop the app to prevent further errors
except Exception as e:
    # Catch any other exception (e.g., file not found, permissions)
    st.error(f"An unexpected error occurred during model loading: {e}")
    st.stop()


# Streamlit
st.title('Bank Customer Churn Predictor')
st.markdown("By Iceyisaak | **[Repo on GitHub 💼](https://github.com/iceyisaak/ann-bank-customer-churn-prediction)**.")

# data_input = {
#  'gender':'Female',
#  'age':45,
#  'balance':200000,
#  'num_of_products':4,
#  'is_active':0,
#  'complained':1,
#  'country':'Germany'
# }


# UIs
country = st.selectbox('Country',oh_encoder.categories_[0])
gender = st.selectbox('Gender',lbl_encoder.classes_)
age = st.slider('Age', 18,99)
balance = st.number_input('Balance')
num_of_products = st.slider('Number of Products',1,4)
is_active = st.selectbox('Is Active Member',[0,1])
complained = st.selectbox('Complained',[0,1])


# Prep Input Data
data_input = pd.DataFrame({
 'gender':[lbl_encoder.transform([gender])],
 'age':[age],
 'balance':[balance],
 'num_of_products':[num_of_products],
 'is_active':[is_active],
 'complained':[complained]
})

# Encode Country input
country_encoded = oh_encoder.transform([[country]])
country_encoded_df = pd.DataFrame(country_encoded, columns=oh_encoder.get_feature_names_out(['country']))

# Concat encoded col with data_input
data_input = pd.concat([data_input.reset_index(drop=True), country_encoded_df], axis=1)

# Scale data_input
data_input_scaled = scaler.transform(data_input)


# Show Prediction
pred = model.predict(data_input_scaled)
pred_proba = pred[0][0]

st.write(f'Churn Probability: {pred_proba: .2f}')

if st.button('Predict'):
    if pred_proba > 0.5:
        st.error('The customer is likely to churn.')
    else:
        st.success('The customer is NOT likely to churn.')
