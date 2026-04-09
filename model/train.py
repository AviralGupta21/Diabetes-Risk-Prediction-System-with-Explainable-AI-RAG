import os
import pickle
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam

np.random.seed(42)
tf.random.set_seed(42)

DATA_PATH = "data/Pima_Indians_Diabetes_Dataset.csv"

df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)

cols_with_zero = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

df[cols_with_zero] = df[cols_with_zero].replace(0, np.nan)
df.fillna(df.median(), inplace=True)

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

feature_names = X.columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\n Training Neural Network...")

nn_model = Sequential([
    Input(shape = (8,)), 
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

nn_model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

nn_model.fit(
    X_train_scaled, y_train,
    validation_data=(X_test_scaled, y_test),
    epochs=50,
    batch_size=16,
    verbose=0
)

nn_probs = nn_model.predict(X_test_scaled).flatten()
nn_preds = (nn_probs > 0.5).astype(int)

nn_acc = accuracy_score(y_test, nn_preds)
print(f"\n Neural Network Accuracy: {nn_acc:.4f}")

print("\n Neural Network Report: ")
print(classification_report(y_test, nn_preds))

print("\n Training Random Forest...")

rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=6,
    random_state=42
)

rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)

rf_acc = accuracy_score(y_test, rf_preds)
print(f" Random Forest Accuracy: {rf_acc:.4f}")

print("\n Training XGBoost...")

xgb_model = XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric='logloss',
    random_state=42
)

xgb_model.fit(X_train, y_train)
xgb_preds = xgb_model.predict(X_test)

xgb_acc = accuracy_score(y_test, xgb_preds)
print(f" XGBoost Accuracy: {xgb_acc:.4f}")

print("\n MODEL COMPARISON")
print("-" * 40)
print(f"Neural Network : {nn_acc:.4f}")
print(f"Random Forest  : {rf_acc:.4f}")
print(f"XGBoost        : {xgb_acc:.4f}")
print("-" * 40)

print("\n Saving Neural Network model and preprocessing artifacts...")

SAVE_DIR = "model"

os.makedirs(SAVE_DIR, exist_ok=True)

nn_model.save(os.path.join(SAVE_DIR, "model.keras"))

with open(os.path.join(SAVE_DIR, "scaler.pkl"), "wb") as f:
    pickle.dump(scaler, f)

with open(os.path.join(SAVE_DIR, "feature_names.pkl"), "wb") as f:
    pickle.dump(feature_names, f)

print(" Saved: model.keras, scaler.pkl, feature_names.pkl")