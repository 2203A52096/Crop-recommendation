import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Streamlit Page Config
st.set_page_config(page_title="Crop Recommendation System", layout="centered")

# Load the trained model
@st.cache_resource
def load_model():
    with open("crop_model.pkl", "rb") as file:
        model = pickle.load(file)
    return model

@st.cache_data
def load_data():
    return pd.read_csv("Crop_recommendation.csv")

# Manual label mapping (int to crop name)
label_map = {
    0: 'apple', 1: 'banana', 2: 'blackgram', 3: 'chickpea', 4: 'coffee',
    5: 'cotton', 6: 'grapes', 7: 'jute', 8: 'kidneybeans', 9: 'lentil',
    10: 'maize', 11: 'mango', 12: 'mothbeans', 13: 'mungbean', 14: 'muskmelon',
    15: 'orange', 16: 'papaya', 17: 'pigeonpeas', 18: 'pomegranate',
    19: 'rice', 20: 'watermelon', 21: 'coconut'
}

# Load resources
model = load_model()
df = load_data()

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Predict Crop", "Data Info"])

# ------------- HOME -------------
if page == "Home":
    st.title("🌾 Crop Recommendation System")
    st.markdown("""
Welcome to the **Crop Recommendation System** built using **Machine Learning**!  
This system suggests the most suitable crop to cultivate based on environmental and soil conditions.

**Model Inputs:**
- Nitrogen (N)
- Phosphorus (P)
- Potassium (K)
- Temperature (°C)
- Humidity (%)
- pH
- Rainfall (mm)

👉 Go to **Predict Crop** tab to try it out!
""")

# ------------- PREDICT CROP -------------
elif page == "Predict Crop":
    st.title("🌱 Predict the Best Crop")

    # Input fields
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
            crop_index = prediction[0]
            crop_name = label_map.get(crop_index, "Unknown")
            st.success(f"✅ Recommended Crop: **{crop_name.capitalize()}**")
        except Exception as e:
            st.error("❌ Error during prediction.")
            st.exception(e)

# ------------- DATA INFO -------------
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
