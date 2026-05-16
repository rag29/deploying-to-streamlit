import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

st.set_page_config(layout="wide")

@st.cache_data
def load_and_train_model():
    '''
    This function simulates loading and training a model.
    In a real app, you would load a pre-trained .pkl or .joblib file here.
    '''
    data = {
        'SquareFeet' : [1500, 2000, 2500, 3000, 1200, 1800, 2200, 2800, 3500, 1600],
        'Bedrooms' : [3,4,3,5,2,3,4,4,5,3],
        'PriceCategory' : ['Low', 'Low', 'High', 'High', 'Low', 'Low', 'High', 'High', 'High', 'Low']
    }
    df = pd.DataFrame(data)
    X = df[['SquareFeet', 'Bedrooms']]
    y = df['PriceCategory']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = LogisticRegression()
    model.fit(X_scaled, y)
    return model, scaler 

model, scaler = load_and_train_model()

st.title("House Price Prediction App")

st.write("Enter the details of a house and get an instant price prediction.")

square_feet = st.slider("Square Footage (sq ft)", min_value=500, max_value=5000, value=2500)
bedrooms = st.slider("Number of Bedrooms", min_value=1, max_value=6, value=3)

if st.button("Get Prediction"):
    user_input = np.array([[square_feet, bedrooms]])
    scaled_input = scaler.transform(user_input)
    prediction = model.predict(scaled_input)[0]
    if prediction == 'High':
        st.success(f"The predicted price category for this house is **{prediction}**")
    else:
        st.info(f"The predicted price category for this house is: **{prediction}**")