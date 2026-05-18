
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

#  Page config 
st.set_page_config(
    page_title="Heart Disease Risk Predictor",
    page_icon="❤️",
    layout="centered",
)

#  Load artifacts 
@st.cache_resource
def load_artifacts():
    model   = joblib.load("model_01_KNN_heart.pkl")
    scaler  = joblib.load("scaler.pkl")
    columns = joblib.load("columns.pkl")   # list of feature names used during training
    return model, scaler, columns

try:
    model, scaler, feature_columns = load_artifacts()
except FileNotFoundError:
    st.error(
        "Model files not found. Run `python train_model.py` first to generate "
        "`model_01_KNN_heart.pkl`, `scaler.pkl`, and `columns.pkl`."
    )
    st.stop()

#  Title 
st.title("❤️ Heart Disease Risk Predictor")
st.markdown(
    "Enter patient details below. The model will estimate whether the patient "
    "is at **high** or **low** risk of heart disease."
)
st.divider()

#  Sidebar: model info 
with st.sidebar:
    st.header("ℹ️ Model Info")
    st.markdown(
        "**Algorithm:** K-Nearest Neighbours (KNN)\n\n"
        "**Dataset:** Heart Failure Prediction (Kaggle)\n\n"
        "**Features:** Age, Sex, Chest Pain Type, Resting BP, Cholesterol, "
        "Fasting Blood Sugar, Resting ECG, Max Heart Rate, Exercise Angina, "
        "Oldpeak, ST Slope\n\n"
        "**Fix applied:** The original pipeline cast scaled features to `int`, "
        "collapsing all values to 0/1 and making the model always predict 'Low Risk'. "
        "This version keeps floats intact."
    )

#  Input form 
st.subheader("Patient Information")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age (years)", min_value=1, max_value=120, value=50)
    sex = st.selectbox("Sex", ["Male (M)", "Female (F)"])
    chest_pain = st.selectbox(
        "Chest Pain Type",
        ["ATA – Atypical Angina", "NAP – Non-Anginal Pain",
         "ASY – Asymptomatic", "TA – Typical Angina"],
    )
    resting_bp = st.number_input(
        "Resting Blood Pressure (mm Hg)", min_value=50, max_value=250, value=120
    )
    cholesterol = st.number_input(
        "Serum Cholesterol (mm/dl)", min_value=0, max_value=600, value=200
    )
    fasting_bs = st.selectbox(
        "Fasting Blood Sugar > 120 mg/dl",
        ["No (0)", "Yes (1)"],
    )

with col2:
    resting_ecg = st.selectbox(
        "Resting ECG",
        ["Normal", "ST – ST-T wave abnormality", "LVH – Left ventricular hypertrophy"],
    )
    max_hr = st.number_input(
        "Maximum Heart Rate Achieved", min_value=40, max_value=250, value=150
    )
    exercise_angina = st.selectbox("Exercise-Induced Angina", ["No (N)", "Yes (Y)"])
    oldpeak = st.number_input(
        "Oldpeak (ST depression)", min_value=-5.0, max_value=10.0,
        value=0.0, step=0.1, format="%.1f"
    )
    st_slope = st.selectbox(
        "ST Slope",
        ["Up – Upsloping", "Flat", "Down – Downsloping"],
    )

st.divider()

#  Preprocessing helper 
def build_input_df(
    age, sex, chest_pain, resting_bp, cholesterol,
    fasting_bs, resting_ecg, max_hr, exercise_angina, oldpeak, st_slope
):
    """
    Reproduce the get_dummies(drop_first=True) encoding used in training.
    The notebook used:
        pd.get_dummies(df, drop_first=True)
    on columns: Sex, ChestPainType, RestingECG, ExerciseAngina, ST_Slope

    drop_first removes the first alphabetical category from each group.
    Surviving dummies (alphabetical order, first dropped):
        Sex      : F dropped → Sex_M
        ChestPain: ASY dropped → ChestPainType_ATA, _NAP, _TA
        RestingECG: LVH dropped → RestingECG_Normal, _ST
        ExAngina : N dropped → ExerciseAngina_Y
        ST_Slope : Down dropped → ST_Slope_Flat, _Up
    """
    sex_val       = sex.split()[0]          # "Male" or "Female"
    cp_val        = chest_pain.split()[0]   # "ATA","NAP","ASY","TA"
    ecg_val       = resting_ecg.split()[0]  # "Normal","ST","LVH"
    ea_val        = "Y" if exercise_angina.startswith("Yes") else "N"
    slope_val     = st_slope.split()[0]     # "Up","Flat","Down"
    fbs_val       = 1 if fasting_bs.startswith("Yes") else 0

    row = {
        "Age":                    age,
        "RestingBP":              resting_bp,
        "Cholesterol":            cholesterol,
        "FastingBS":              fbs_val,
        "MaxHR":                  max_hr,
        "Oldpeak":                oldpeak,
        # One-hot dummies (drop_first=True encoding)
        "Sex_M":                  1 if sex_val == "Male" else 0,
        "ChestPainType_ATA":      1 if cp_val == "ATA" else 0,
        "ChestPainType_NAP":      1 if cp_val == "NAP" else 0,
        "ChestPainType_TA":       1 if cp_val == "TA" else 0,
        "RestingECG_Normal":      1 if ecg_val == "Normal" else 0,
        "RestingECG_ST":          1 if ecg_val == "ST" else 0,
        "ExerciseAngina_Y":       1 if ea_val == "Y" else 0,
        "ST_Slope_Flat":          1 if slope_val == "Flat" else 0,
        "ST_Slope_Up":            1 if slope_val == "Up" else 0,
    }

    df = pd.DataFrame([row])

    # Align to the exact column order used during training
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_columns]

    return df


#  Predict 
if st.button("🔍 Predict Risk", use_container_width=True, type="primary"):
    input_df = build_input_df(
        age, sex, chest_pain, resting_bp, cholesterol,
        fasting_bs, resting_ecg, max_hr, exercise_angina, oldpeak, st_slope
    )

    # Scale (floats stay as floats — this is the critical fix)
    input_scaled = scaler.transform(input_df)

    prediction   = model.predict(input_scaled)[0]
    probability  = model.predict_proba(input_scaled)[0]

    low_prob  = probability[0] * 100
    high_prob = probability[1] * 100

    st.divider()
    st.subheader("Prediction Result")

    if prediction == 1:
        st.error(f"⚠️ **High Risk of Heart Disease** ({high_prob:.1f}% confidence)")
        st.markdown(
            "> The model indicates this patient may be at **high risk**. "
            "Please consult a cardiologist for further evaluation."
        )
    else:
        st.success(f"✅ **Low Risk of Heart Disease** ({low_prob:.1f}% confidence)")
        st.markdown(
            "> The model indicates this patient appears to be at **low risk**. "
            "Continue with regular health check-ups."
        )

    # Probability bar
    st.markdown("**Prediction Probabilities**")
    prob_df = pd.DataFrame(
        {"Risk Level": ["Low Risk", "High Risk"],
         "Probability (%)": [round(low_prob, 2), round(high_prob, 2)]}
    )
    st.bar_chart(prob_df.set_index("Risk Level"))

    # Show processed input for transparency
    with st.expander("🔬 See processed input features"):
        st.dataframe(input_df)

st.caption(
    "Disclaimer: This tool is for educational purposes only and is "
    "not a substitute for professional medical advice."
)
