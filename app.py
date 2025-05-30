import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

from sklearn.preprocessing import LabelEncoder

# Streamlit Page Configuration
st.set_page_config(page_title="Crop Recommendation System", layout="centered")

# Load model and label encoder
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as file:
        model = pickle.load(file)
    return model

@st.cache_resource
def load_label_encoder():
    with open("label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)
    return label_encoder

@st.cache_data
def load_data():
    return pd.read_csv("Crop_recommendation.csv")

# Load assets
model = load_model()
label_encoder = load_label_encoder()
df = load_data()

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Predict Crop", "Data Info"])

# -------------------- HOME PAGE --------------------
if page == "Home":
    st.title("🌾 Crop Recommendation System")
    st.markdown("""
Welcome to the **Crop Recommendation System** built using **Machine Learning**!  
This system recommends the most suitable crop to grow based on soil and climate conditions.

**Features used:**
- Nitrogen (N)
- Phosphorus (P)
- Potassium (K)
- Temperature (°C)
- Humidity (%)
- pH
- Rainfall (mm)

👉 Use the **'Predict Crop'** tab to test it out!
""")

# -------------------- PREDICTION PAGE --------------------
elif page == "Predict Crop":
    st.title("🌱 Predict the Best Crop")

    # Input features
    n = st.number_input("Nitrogen content (N)", 0.0, 140.0, 50.0)
    p = st.number_input("Phosphorus content (P)", 5.0, 145.0, 50.0)
    k = st.number_input("Potassium content (K)", 5.0, 205.0, 50.0)
    temperature = st.number_input("Temperature (°C)", 8.0, 45.0, 25.0)
    humidity = st.number_input("Humidity (%)", 10.0, 100.0, 60.0)
    ph = st.number_input("Soil pH", 3.5, 9.5, 6.5)
    rainfall = st.number_input("Rainfall (mm)", 20.0, 300.0, 100.0)

    if st.button("Recommend Crop"):
        try:
            input_data = np.array([[n, p, k, temperature, humidity, ph, rainfall]])
            prediction = model.predict(input_data)
            predicted_crop = label_encoder.inverse_transform(prediction)[0]
            st.success(f"✅ Recommended Crop: **{predicted_crop.capitalize()}**")
        except Exception as e:
            st.error("❌ Error during prediction.")
            st.exception(e)

# -------------------- DATA INFO PAGE --------------------
elif page == "Data Info":
    st.title("📊 Dataset Overview")

    st.subheader("🔍 Sample Data")
    st.dataframe(df.head())

    st.subheader("📈 Summary Statistics")
    st.write(df.describe())

    st.subheader("📌 Crop Distribution")
    st.write(df['label'].value_counts())

    st.subheader("📦 Dataset Shape")
    st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
