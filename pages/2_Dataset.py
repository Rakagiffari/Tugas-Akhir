import streamlit as st
import pandas as pd

st.title("🗂️ Dataset")

df = pd.read_csv("data/data.csv")

st.subheader("Preview Dataset")
st.dataframe(df)

st.subheader("Missing Value")

missing = df.isnull().sum()

st.dataframe(missing)
