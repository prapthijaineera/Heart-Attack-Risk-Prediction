import pickle
import streamlit as st
import pandas as pd

st.title("Heart Attack Risk Classification")

# Load trained model
model = pickle.load(open("gboost_model.pkl", "rb"))

# -------------------------------
# Input Features
# -------------------------------

Age = st.number_input("Age", min_value=20, max_value=100, value=25)

RestingBP = st.number_input(
    "RestingBP", min_value=0, max_value=250, value=120
)

Cholesterol = st.number_input(
    "Cholesterol", min_value=0, max_value=650, value=200
)

MaxHR = st.number_input(
    "MaxHR", min_value=60, max_value=250, value=150
)

Oldpeak = st.number_input(
    "Oldpeak", min_value=-3.0, max_value=7.0, value=1.0
)

FastingBS = st.selectbox(
    "FastingBS", [0, 1]
)

Gender = st.selectbox(
    "Gender", ["M", "F"]
)

ChestPainType = st.selectbox(
    "ChestPainType",
    ["ATA", "NAP", "ASY", "TA"]
)

ExerciseAngina = st.selectbox(
    "ExerciseAngina",
    ["Y", "N"]
)

RestingECG = st.selectbox(
    "RestingECG",
    ["Normal", "ST", "LVH"]
)

ST_Slope = st.selectbox(
    "ST_Slope",
    ["Up", "Flat", "Down"]
)

# -------------------------------
# Encoding
# -------------------------------

# Gender
sex = 1 if Gender == "M" else 0

# Exercise Angina
exerciseAngina = 1 if ExerciseAngina == "Y" else 0

# ChestPainType One-Hot Encoding
ChestPainType_ASY = 1 if ChestPainType == "ASY" else 0
ChestPainType_ATA = 1 if ChestPainType == "ATA" else 0
ChestPainType_NAP = 1 if ChestPainType == "NAP" else 0
ChestPainType_TA = 1 if ChestPainType == "TA" else 0

# RestingECG One-Hot Encoding
RestingECG_LVH = 1 if RestingECG == "LVH" else 0
RestingECG_Normal = 1 if RestingECG == "Normal" else 0
RestingECG_ST = 1 if RestingECG == "ST" else 0

# ST_Slope Encoding
# Mapping used during training:
# Up = 0, Down = 1, Flat = 2

st_slope_dict = {
    "Up": 0,
    "Down": 1,
    "Flat": 2
}

st_Slope = st_slope_dict[ST_Slope]

# -------------------------------
# Create Input DataFrame
# -------------------------------

input_features = pd.DataFrame({
    "Age": [Age],
    "RestingBP": [RestingBP],
    "Cholesterol": [Cholesterol],
    "FastingBS": [FastingBS],
    "MaxHR": [MaxHR],
    "Oldpeak": [Oldpeak],
    "sex": [sex],
    "exerciseAngina": [exerciseAngina],

    "RestingECG_LVH": [RestingECG_LVH],
    "RestingECG_Normal": [RestingECG_Normal],
    "RestingECG_ST": [RestingECG_ST],

    "ChestPainType_ASY": [ChestPainType_ASY],
    "ChestPainType_ATA": [ChestPainType_ATA],
    "ChestPainType_NAP": [ChestPainType_NAP],
    "ChestPainType_TA": [ChestPainType_TA],

    "st_Slope": [st_Slope]
})

# -------------------------------
# Prediction
# -------------------------------

if st.button("Predict"):

    prediction = model.predict(input_features)

    if prediction[0] == 1:
        st.error("High Risk of Heart Attack")
    else:
        st.success("Low Risk of Heart Attack")
