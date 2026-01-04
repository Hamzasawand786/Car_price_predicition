# app.py

import streamlit as st
import pandas as pd
import joblib
import os

# -----------------------------
# Streamlit page configuration
# -----------------------------
st.set_page_config(
    page_title="🏎️ Sports Car Price Predictor",
    page_icon="🚗",
    layout="centered"
)

# -----------------------------
# Custom CSS for sporty look
# -----------------------------
st.markdown("""
    <style>
    .main {
        background-color: #1e1e1e;
        color: #f0f0f0;
    }
    h1 {
        color: #ff4b4b;
        text-align: center;
    }
    .stButton>button {
        background-color: #ff4b4b;
        color: white;
        font-size: 16px;
        border-radius: 10px;
        padding: 10px;
        width: 100%;
    }
    .stTextInput>div>input {
        border-radius: 8px;
        padding: 8px;
        background-color: #2b2b2b;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------------
# Title
# -----------------------------
st.title("🏎️ Sports Car Price Predictor")

# -----------------------------
# Model upload section
# -----------------------------
st.sidebar.header("Upload Model")
uploaded_model = st.sidebar.file_uploader(
    "Upload your trained car price model (.pkl file)",
    type=["pkl"]
)

# -----------------------------
# Check if model is uploaded
# -----------------------------
if uploaded_model is not None:
    # Save uploaded model to temp file
    model_path = "uploaded_car_model.pkl"
    with open(model_path, "wb") as f:
        f.write(uploaded_model.getbuffer())
    
    # Load model
    model = joblib.load(model_path)
    
    st.success("✅ Model loaded successfully!")
    
    # -----------------------------
    # Input form
    # -----------------------------
    st.subheader("Enter Car Details:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        year = st.number_input("Year", min_value=1980, max_value=2026, value=2020)
        mileage = st.number_input("Mileage (in km)", min_value=0, value=10000)
        engine_size = st.number_input("Engine Size (L)", min_value=0.5, max_value=8.0, value=3.0, step=0.1)
    
    with col2:
        fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "Electric", "Hybrid"])
        transmission = st.selectbox("Transmission", ["Manual", "Automatic"])
        brand = st.text_input("Brand", value="Ferrari")
    
    # -----------------------------
    # Prediction button
    # -----------------------------
    if st.button("Predict Price"):
        input_data = pd.DataFrame({
            "year": [year],
            "mileage": [mileage],
            "engine_size": [engine_size],
            "fuel_type": [fuel_type],
            "transmission": [transmission],
            "brand": [brand]
        })
        
        # Make prediction
        price = model.predict(input_data)[0]
        
        st.markdown(f"""
            <div style='background-color:#ff4b4b; padding:15px; border-radius:10px; text-align:center;'>
                <h2 style='color:white;'>Estimated Price: 💰 ${price:,.2f}</h2>
            </div>
        """, unsafe_allow_html=True)
    
else:
    st.warning("⚠️ Please upload your trained car price model (.pkl file) from the sidebar to start predicting.")
