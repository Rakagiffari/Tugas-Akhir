import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# LOAD DATA
df = pd.read_csv("data/data.csv")

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

# HAPUS DATA KOSONG
df.dropna(inplace=True)

# TARGET
target_column = "Service"

# ENCODING
label_encoders = {}

for col in df.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# SPLIT X Y
X = df.drop(columns=[target_column])
y = df[target_column]

# SPLIT TRAIN TEST
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# MODEL
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# PREDIKSI
y_pred = model.predict(X_test)

# EVALUASI
accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy :", accuracy)
print("\nClassification Report :")
print(classification_report(y_test, y_pred))

# SIMPAN MODEL
joblib.dump(model, "model/random_forest_model.pkl")
joblib.dump(label_encoders, "model/label_encoders.pkl")
joblib.dump(X.columns.tolist(), "model/feature_columns.pkl")

print("\nModel berhasil disimpan!")
