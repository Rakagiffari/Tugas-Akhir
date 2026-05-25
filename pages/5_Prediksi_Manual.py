import streamlit as st
import pandas as pd
import joblib

st.title("🔍 Prediksi Manual")

model = joblib.load("model/random_forest_model.pkl")
feature_columns = joblib.load("model/feature_columns.pkl")

input_data = {}

for col in feature_columns:

    value = st.number_input(
        f"Input {col}",
        value=0.0
    )

    input_data[col] = value

if st.button("Prediksi"):

    input_df = pd.DataFrame([input_data])

    prediction = model.predict(input_df)[0]

    st.success(f"Hasil Prediksi Service : {prediction}")
