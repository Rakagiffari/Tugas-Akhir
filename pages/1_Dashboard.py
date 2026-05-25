import streamlit as st
import pandas as pd
import plotly.express as px

st.title("📊 Dashboard")

df = pd.read_csv("data/data.csv")

col1, col2, col3 = st.columns(3)

col1.metric("Jumlah Data", len(df))
col2.metric("Jumlah Kolom", len(df.columns))

if 'Service' in df.columns:
    col3.metric("Jumlah Kelas", df['Service'].nunique())

st.divider()

if 'Service' in df.columns:
    fig = px.pie(
        df,
        names='Service',
        title='Distribusi Service'
    )

    st.plotly_chart(fig, use_container_width=True)
