import streamlit as st
import pandas as pd
import joblib

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score
)

from sklearn.model_selection import train_test_split

st.title("🧠 Evaluasi Model")

# LOAD
model = joblib.load("model/random_forest_model.pkl")
label_encoders = joblib.load("model/label_encoders.pkl")

# DATA
df = pd.read_csv("data/data.csv")

df.dropna(inplace=True)

target_column = "Service"

# ENCODING
for col in df.select_dtypes(include='object').columns:

    if col in label_encoders:

        le = label_encoders[col]

        df[col] = le.transform(df[col].astype(str))

# SPLIT
X = df.drop(columns=[target_column])
y = df[target_column]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# PREDICT
y_pred = model.predict(X_test)

# ACCURACY
acc = accuracy_score(y_test, y_pred)

st.metric("Accuracy", f"{acc:.2%}")

# REPORT
st.subheader("Classification Report")

report = classification_report(
    y_test,
    y_pred,
    output_dict=True
)

st.dataframe(
    pd.DataFrame(report).transpose()
)

# CONFUSION MATRIX
st.subheader("Confusion Matrix")

cm = confusion_matrix(y_test, y_pred)

st.write(cm)
