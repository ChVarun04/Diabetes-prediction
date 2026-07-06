import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Diabetes Prediction System",
    page_icon="🩺",
    layout="wide"
)

import joblib

import os

st.write("Current directory:", os.getcwd())
st.write("Files in directory:", os.listdir())

try:
    model = joblib.load("diabetes_pipeline.pkl")
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()


st.markdown("""
<style>
.main{
    background-color:#f7f9fc;
}

.stButton>button{
    width:100%;
    background-color:#0E8A16;
    color:white;
    border-radius:10px;
    font-size:18px;
    height:3em;
}

.stButton>button:hover{
    background-color:#056608;
    color:white;
}

h1{
    color:#0E8A16;
}

.block-container{
    padding-top:2rem;
}

</style>
""", unsafe_allow_html=True)


st.sidebar.title("🩺 Diabetes Prediction")

st.sidebar.info("""
This application predicts whether a patient is likely to have diabetes based on medical information.

**Machine Learning Model**

✔ Preprocessing Pipeline

✔ One-Hot Encoding

✔ Feature Scaling

✔ Classification Model
""")

st.sidebar.success("Developed using Streamlit")


st.title("🩺 Diabetes Prediction System")

st.write(
    "Enter the patient's medical details below and click **Predict**."
)

st.divider()



col1, col2 = st.columns(2)

with col1:

    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    age = st.number_input(
        "Age",
        min_value=1,
        max_value=120,
        value=30
    )

    hypertension = st.selectbox(
        "Hypertension",
        [0,1],
        format_func=lambda x:"Yes" if x==1 else "No"
    )

    heart_disease = st.selectbox(
        "Heart Disease",
        [0,1],
        format_func=lambda x:"Yes" if x==1 else "No"
    )

with col2:

    smoking_history = st.selectbox(
        "Smoking History",
        [
            "never",
            "former",
            "current",
            "not current",
            "ever",
            "No Info"
        ]
    )

    bmi = st.number_input(
        "BMI",
        min_value=10.0,
        max_value=70.0,
        value=25.0
    )

    HbA1c_level = st.number_input(
        "HbA1c Level",
        min_value=3.0,
        max_value=15.0,
        value=5.5
    )

    blood_glucose_level = st.number_input(
        "Blood Glucose Level",
        min_value=50,
        max_value=350,
        value=120
    )



input_df = pd.DataFrame({

    "gender":[gender],
    "age":[age],
    "hypertension":[hypertension],
    "heart_disease":[heart_disease],
    "smoking_history":[smoking_history],
    "bmi":[bmi],
    "HbA1c_level":[HbA1c_level],
    "blood_glucose_level":[blood_glucose_level]

})

st.write("### Input Summary")
st.dataframe(input_df, use_container_width=True)

if st.button("Predict Diabetes"):

    prediction = model.predict(input_df)[0]

    st.divider()

    if prediction == 1:

        st.error("⚠️ High Risk of Diabetes")

    else:

        st.success("✅ Low Risk of Diabetes")

    if hasattr(model, "predict_proba"):

        probability = model.predict_proba(input_df)[0][prediction]

        st.metric(
            "Prediction Confidence",
            f"{probability*100:.2f}%"
        )
