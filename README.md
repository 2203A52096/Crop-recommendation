# 🌾 Crop Recommendation System

The dataset is taken from kaggle from https://www.kaggle.com/datasets/madhuraatmarambhagat/crop-recommendation-dataset

## 📝 Problem Statement

Farmers often struggle to select the most suitable crop for cultivation due to changing soil and environmental conditions. This project builds a **machine learning model** that recommends the most appropriate crop based on key parameters like soil nutrients, temperature, humidity, pH, and rainfall.

The goal is to help farmers make **data-driven decisions** to improve productivity and sustainability.

---

## 📊 Dataset Description

The dataset used is _Crop_recommendation.csv_ and contains the following columns:

### Features (Inputs):

* **N**: Nitrogen content in soil
* **P**: Phosphorus content in soil
* **K**: Potassium content in soil
* **temperature**: Temperature in °C
* **humidity**: Relative humidity in %
* **ph**: pH value of the soil
* **rainfall**: Rainfall in mm

### Target (Output):

* **label**: The recommended crop to grow (e.g., rice, maize, chickpea, etc.)

---

## 🚀 Project Workflow

### 1. **Data Reading**

* Loaded the dataset using Pandas.

### 2. **Data Evaluation**

* Displayed basic statistics and checked for null or missing values.
* Confirmed data types and feature ranges.

### 3. **Data Visualization**

* Used **box plots** to visualize each feature and identify **outliers**.

### 4. **Feature Importance**

* Used **Random Forest** to identify which features most influence crop recommendation.

### 5. **Label Encoding**

* Converted the crop labels (strings) into numeric format using _LabelEncoder_.

### 6. **Feature and Target Separation**

* Split the dataset into:

  * _X_ → input features: \[N, P, K, temperature, humidity, ph, rainfall]
  * _y_ → target crop label

### 7. **Outlier Removal**

* Removed extreme values based on box plot analysis to improve model robustness.

### 8. **Apply SMOTE**

* Balanced the dataset using **SMOTE** to generate synthetic examples of minority crop classes.

### 9. **Preprocessing**

* Applied **standard scaling** to normalize feature values, which improves model performance (especially for SVM).

### 10. **Model Training**

* Trained the following models:

  * **SVM (Support Vector Machine)**
  * **Random Forest Classifier**

### 11. **Cross-Validation**

* Performed 5-fold cross-validation for both models:

  * **SVM Average Accuracy:** 97.3%
  * **Random Forest Average Accuracy:** 99.4% ✅ (Best Model)

### 12. **Model Saving**

* Saved the best-performing model (**Random Forest**) using _pickle_ as _crop_model.pkl_.

---

## Solution

**Crop Recommendation System**

An app that helps farmers or users find the best crop to grow based on soil nutrients and weather conditions. It uses a machine learning model to predict the most suitable crop by analyzing inputs like nitrogen, phosphorus, potassium levels, temperature, humidity, soil pH, and rainfall.

The app has three main pages:

1. **Home:** Introduces the system and explains what features are used for crop prediction.
2. **Predict Crop:** Allows users to input soil and climate data to get a recommended crop.
3. **Data Info:** Shows an overview of the dataset used to train the model, including sample data, statistics, and crop distribution.

