import streamlit as st
import pandas as pd
import joblib

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

st.title("🧠 Evaluasi Model")

df = pd.read_csv("data/data.csv")

model = joblib.load("model/random_forest_model.pkl")

target_column = "Service"

df.dropna(inplace=True)

X = df.drop(columns=[target_column])
y = df[target_column]

X = pd.get_dummies(X)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

y_pred = model.predict(X_test)

st.subheader("Classification Report")

report = classification_report(y_test, y_pred, output_dict=True)

st.dataframe(pd.DataFrame(report).transpose())

st.subheader("Confusion Matrix")

cm = confusion_matrix(y_test, y_pred)

st.write(cm)
