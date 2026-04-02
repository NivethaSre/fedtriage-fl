import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle
import os

FEATURE_COLUMNS = [
    "age", "gender",
    "diabetes", "hypertension", "immunocompromised",
    "fever", "cough", "shortness_of_breath", "chest_pain",
    "headache", "fatigue", "nausea", "vomiting",
    "diarrhea", "rash", "joint_pain", "loss_of_smell",
    "travel_history",
    "temperature_celsius", "heart_rate", "oxygen_saturation",
    "systolic_bp", "symptom_duration_days",
]

TARGET_COLUMN = "triage_label"


def preprocess(df, scaler=None, fit_scaler=True):
    df = df.copy()
    df["gender"] = (df["gender"] == "M").astype(int)
    travel_map = {"none": 0, "domestic": 1, "europe": 2,
                  "southeast_asia": 3, "middle_east": 4, "africa": 5}
    df["travel_history"] = df["travel_history"].map(travel_map).fillna(0).astype(int)

    X = df[FEATURE_COLUMNS].values.astype(np.float32)
    y = df[TARGET_COLUMN].values.astype(np.int64)

    continuous_cols = [
        FEATURE_COLUMNS.index(c) for c in [
            "age", "temperature_celsius", "heart_rate",
            "oxygen_saturation", "systolic_bp", "symptom_duration_days"
        ]
    ]

    if fit_scaler:
        if scaler is None:
            scaler = StandardScaler()
        X[:, continuous_cols] = scaler.fit_transform(X[:, continuous_cols])
    else:
        X[:, continuous_cols] = scaler.transform(X[:, continuous_cols])

    return X, y, scaler


def load_clinic_data(clinic_id, data_dir="data/raw", test_size=0.2):
    path = os.path.join(data_dir, f"{clinic_id}.csv")
    df = pd.read_csv(path)
    X, y, scaler = preprocess(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test, scaler


def save_scaler(scaler, path="models/saved/scaler.pkl"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(scaler, f)


def load_scaler(path="models/saved/scaler.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)