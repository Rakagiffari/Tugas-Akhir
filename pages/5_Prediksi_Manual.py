import streamlit as st
import pandas as pd
import joblib

st.title("🔍 Prediksi Manual")

# LOAD
model = joblib.load("model/random_forest_model.pkl")
label_encoders = joblib.load("model/label_encoders.pkl")
feature_columns = joblib.load("model/feature_columns.pkl")

input_data = {}

for col in feature_columns:

    if col in label_encoders:

        options = list(label_encoders[col].classes_)

        selected = st.selectbox(
            f"Pilih {col}",
            options
        )

        encoded = label_encoders[col].transform([selected])[0]

        input_data[col] = encoded

    else:

        value = st.number_input(
            f"Input {col}",
            value=0.0
        )

        input_data[col] = value

if st.button("Prediksi"):

    input_df = pd.DataFrame([input_data])

    prediction = model.predict(input_df)[0]

    probabilities = model.predict_proba(input_df)

    confidence = probabilities.max() * 100

    st.success(f"""
    Prediksi Service : {prediction}

    Confidence : {confidence:.2f}%
    """)
