import pickle
import numpy as np
import pandas as pd
import shap
import tensorflow as tf
import os

_MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR  = os.path.dirname(_MODEL_DIR)

model = tf.keras.models.load_model(os.path.join(_MODEL_DIR, "model.keras"))

with open(os.path.join(_MODEL_DIR, "scaler.pkl"), "rb") as f:
    scaler = pickle.load(f)

with open(os.path.join(_MODEL_DIR, "feature_names.pkl"), "rb") as f:
    feature_names = pickle.load(f)

DATA_PATH = os.path.join(_ROOT_DIR, "data", "Pima_Indians_Diabetes_Dataset.csv")
df = pd.read_csv(DATA_PATH)

cols_with_zero = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df[cols_with_zero] = df[cols_with_zero].replace(0, np.nan)
df.fillna(df.median(), inplace=True)

X = df.drop("Outcome", axis=1)

background_data = scaler.transform(X.sample(100, random_state=42))
explainer = shap.GradientExplainer(model, background_data)

ACTIONABLE_FEATURES = {
    "Glucose", "BMI", "Insulin",
    "BloodPressure", "SkinThickness", "Pregnancies"
}

NON_ACTIONABLE_FEATURES = {
    "DiabetesPedigreeFunction", "Age"
}


def explain_instance(input_dict: dict) -> dict:

    input_df = pd.DataFrame([input_dict])[feature_names]

    cols_with_zero = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    input_df[cols_with_zero] = input_df[cols_with_zero].replace(0, np.nan)
    input_df.fillna(df.median(), inplace=True)
    
    input_scaled = scaler.transform(input_df)

    prob = float(model.predict(input_scaled, verbose=0)[0][0])
    prediction = int(prob > 0.5)
    confidence = round(abs(prob - 0.5) * 200, 1)

    shap_values = explainer.shap_values(input_scaled)
    shap_vals = np.array(shap_values[0]).flatten()

    feature_shap = sorted(
        zip(feature_names, shap_vals),
        key=lambda x: abs(x[1]),
        reverse=True
    )

    top_features_display = feature_shap[:5]
    top_features_rag     = feature_shap[:3]

    actionable = [
        (f, v) for f, v in top_features_rag if f in ACTIONABLE_FEATURES
    ]
    non_actionable = [
        (f, v) for f, v in top_features_rag if f in NON_ACTIONABLE_FEATURES
    ]

    return {
        "prediction"             : prediction,
        "probability"            : prob,
        "confidence"             : confidence,
        "top_features"           : top_features_display,
        "top_features_rag"       : top_features_rag,
        "actionable_features"    : actionable,
        "non_actionable_features": non_actionable,
        "all_features"           : feature_shap 
    }


if __name__ == "__main__":
    sample_input = {
        "Pregnancies": 2, "Glucose": 150,
        "BloodPressure": 80, "SkinThickness": 25,
        "Insulin": 100, "BMI": 30,
        "DiabetesPedigreeFunction": 0.5, "Age": 35
    }

    result = explain_instance(sample_input)

    print("\n Prediction  :", "High Risk" if result["prediction"] == 1 else "Low Risk")
    print("Probability :", round(result["probability"], 4))
    print("Confidence  :", result["confidence"], "%")

    print("\n Top 5 Features (displayed to user):")
    for f, v in result["top_features"]:
        print(f"  {f}: {v:.4f}")

    print("\n Top 3 Features (drive RAG query):")
    for f, v in result["top_features_rag"]:
        print(f"  {f}: {v:.4f}")

    print("\n Actionable:")
    for f, v in result["actionable_features"]:
        print(f"  {f}: {v:.4f}")

    print("\n Non-Actionable:")
    for f, v in result["non_actionable_features"]:
        print(f"  {f}: {v:.4f}")

    print("\n All 8 Features:")
    for f, v in result["all_features"]:
        print(f"  {f}: {v:.4f}")