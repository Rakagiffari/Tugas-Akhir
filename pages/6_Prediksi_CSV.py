import streamlit as st
import pandas as pd
import joblib

st.title("📁 Prediksi CSV")

# LOAD MODEL
model = joblib.load("model/random_forest_model.pkl")
label_encoders = joblib.load("model/label_encoders.pkl")
feature_columns = joblib.load("model/feature_columns.pkl")

uploaded_file = st.file_uploader(
    "Upload File CSV",
    type=["csv"]
)

if uploaded_file is not None:

    try:

        # READ CSV
        df = pd.read_csv(uploaded_file)

        st.subheader("Preview Data Upload")
        st.dataframe(df.head())

        # HAPUS KOLOM YANG TIDAK DIPAKAI
        drop_columns = [
            'Nama',
            'KTP',
            'Telepon No',
            'Invoice No',
            'Plate No'
        ]

        for col in drop_columns:
            if col in df.columns:
                df.drop(columns=col, inplace=True)

        # HANDLE MISSING VALUE
        df.fillna(0, inplace=True)

        # ENCODING
        for col in df.columns:

            if col in label_encoders:

                le = label_encoders[col]

                df[col] = df[col].astype(str)

                valid_classes = set(le.classes_)

                df[col] = df[col].apply(
                    lambda x: x if x in valid_classes else le.classes_[0]
                )

                df[col] = le.transform(df[col])

        # SAMAKAN FITUR
        for col in feature_columns:
            if col not in df.columns:
                df[col] = 0

        df = df[feature_columns]

        # PREDIKSI
        predictions = model.predict(df)

        # CONFIDENCE
        probabilities = model.predict_proba(df)
        confidence = probabilities.max(axis=1)

        # HASIL
        result_df = pd.DataFrame()

        result_df["Prediksi_Service"] = predictions
        result_df["Confidence"] = confidence

        final_df = pd.concat([df, result_df], axis=1)

        st.success("Prediksi berhasil dilakukan!")

        st.subheader("Hasil Prediksi")
        st.dataframe(final_df)

        # DOWNLOAD CSV
        csv = final_df.to_csv(index=False)

        st.download_button(
            label="📥 Download Hasil CSV",
            data=csv,
            file_name="hasil_prediksi.csv",
            mime="text/csv"
        )

    except Exception as e:

        st.error(f"Terjadi Error: {e}")
