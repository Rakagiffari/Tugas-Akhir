import streamlit as st
import pandas as pd
import plotly.express as px

st.title("📈 Visualisasi Data")

df = pd.read_csv("data/data.csv")

if 'Last Kilometer' in df.columns and 'Service' in df.columns:

    fig = px.box(
        df,
        x='Service',
        y='Last Kilometer',
        color='Service',
        title='Kilometer vs Service'
    )

    st.plotly_chart(fig, use_container_width=True)

if 'Parts Qty' in df.columns and 'Service' in df.columns:

    fig2 = px.box(
        df,
        x='Service',
        y='Parts Qty',
        color='Service',
        title='Parts Qty vs Service'
    )

    st.plotly_chart(fig2, use_container_width=True)
