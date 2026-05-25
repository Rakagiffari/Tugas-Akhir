import streamlit as st
import pandas as pd
import joblib

st.title("📁 Prediksi CSV")

model = joblib.load("model/random_forest_model.pkl")

uploaded_file = st.file_uploader(
    "Upload File CSV",
    type=['csv']
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    predictions = model.predict(df)

    df['Hasil_Prediksi'] = predictions

    st.dataframe(df)

    csv = df.to_csv(index=False)

    st.download_button(
        label="Download Hasil",
        data=csv,
        file_name='hasil_prediksi.csv',
        mime='text/csv'
    )
