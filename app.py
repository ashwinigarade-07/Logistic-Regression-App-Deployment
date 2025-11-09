import streamlit as st
import pickle
import numpy as np

# Load trained model
model = pickle.load(open('logistic_model.pkl', 'rb'))

# Title
st.title("🩺 Diabetes Prediction using Logistic Regression")
st.write("Enter patient details below to predict the diabetes outcome:")

# Input fields
pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, step=1)
glucose = st.number_input("Glucose Level", min_value=0.0, max_value=300.0, step=1.0)
blood_pressure = st.number_input("Blood Pressure", min_value=0.0, max_value=200.0, step=1.0)
skin_thickness = st.number_input("Skin Thickness", min_value=0.0, max_value=100.0, step=1.0)
insulin = st.number_input("Insulin Level", min_value=0.0, max_value=900.0, step=1.0)
bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, step=0.1)
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, step=0.01)
age = st.number_input("Age", min_value=0, max_value=120, step=1)

# Predict button
if st.button("Predict"):
    features = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                          insulin, bmi, dpf, age]])
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1]

    st.subheader("Prediction Result:")
    if prediction == 1:
        st.error(f"⚠ The model predicts *Diabetic* (probability: {probability:.2f})")
    else:
        st.success(f"The model predicts *Non-Diabetic* (probability: {probability:.2f})")
