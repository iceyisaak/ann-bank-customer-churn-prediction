import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder

import os
# Disables all GPUs, forcing operations onto the CPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from keras.models import load_model
import streamlit as st

# Import model
model = load_model('ann.keras')

# Encoders & Scalers
lb_encoder = LabelEncoder()
oh_encoder = OneHotEncoder()
scaler = StandardScaler()

# Streamlit
st.title('Bank Customer Churn Predictor')

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
gender = st.selectbox('Gender',lb_encoder.classes_)
age = st.slider('Age', 18,99)
balance = st.number_input('Balance')
num_of_products = st.slider('Number of Products',1,4)
is_active = st.selectbox('Is Active Member',[0,1])
complained = st.selectbox('Complained',[0,1])


# Prep Input Data
data_input = pd.DataFrame({
 'gender':[lb_encoder.transform([gender][0])],
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

if pred_proba > 0.5:
    st.write('The customer is likely to churn.')
else:
    st.write('The customer is NOT likely to churn.')
