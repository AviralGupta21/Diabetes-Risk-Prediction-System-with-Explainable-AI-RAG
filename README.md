# 🧠 Diabetes Risk Prediction System with Explainable AI + RAG

An end-to-end **Machine Learning + Explainable AI + Retrieval-Augmented Generation (RAG)** system that predicts diabetes risk and provides **clinically grounded explanations and personalized advice**.

---

## 🚀 Project Overview

This project predicts whether a patient is at **high or low risk of diabetes** using clinical features and enhances the prediction with:

* 🔍 **Explainability (SHAP)** – identifies key contributing factors
* 📚 **RAG (ChromaDB + Medical Corpus)** – retrieves evidence-based context
* 🤖 **LLM (Ollama + Mistral)** – generates human-readable explanations & advice
* 🌐 **Full-stack Web App** – interactive frontend + FastAPI backend

---

## 🧩 System Architecture

```
User Input → ML Model → SHAP Explanation → RAG Retrieval → LLM → Final Output
```

* Model prediction + confidence
* Feature importance (SHAP values)
* Clinical explanation (grounded in literature)
* Personalized lifestyle advice

---

## 📊 Model Performance Comparison

| Model             | Accuracy   |
| ----------------- | ---------- |
| 🧠 Neural Network | **0.8377** |
| 🌲 Random Forest  | **0.8636** |
| ⚡ XGBoost         | **0.8766** |

👉 While **XGBoost achieved the highest accuracy**, the **Neural Network was chosen as the base model** for this project.

### 💡 Why Neural Network?

* ✅ **Better compatibility with SHAP (GradientExplainer)** for deep learning models
* ✅ Provides **smoother, instance-level explanations** compared to tree-based models
* ✅ Easier integration with **end-to-end AI pipelines (TensorFlow ecosystem)**
* ✅ More suitable for **future scalability** (e.g., deep learning extensions, multimodal inputs)
* ✅ Aligns well with **modern AI system design (ML + DL + LLM integration)**

👉 In this project, **interpretability + pipeline compatibility** were prioritized over marginal accuracy gains.

---

## 📁 Project Structure

```
PBL_Project_6th/
│
├── backend/
│   ├── main.py              # FastAPI server
│   └── pipeline.py          # RAG + LLM pipeline
│
├── frontend/
│   └── index.html           # UI dashboard
│
├── model/
│   ├── train.py             # Model training
│   ├── explain.py           # SHAP explanations
│   ├── model.keras
│   ├── scaler.pkl
│   └── feature_names.pkl
│
├── rag/
│   ├── ingest.py            # Build vector DB
│   ├── retriever.py         # Context retrieval
│   ├── corpus/
│   └── chroma_db/
│
├── data/
│   └── Pima Indians Dataset
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the repository

```bash
git clone https://github.com/your-username/diabetes-risk-prediction.git
cd diabetes-risk-prediction
```

---

### 2️⃣ Create virtual environment

```bash
python -m venv .venv
source .venv/bin/activate     # Linux/Mac
.venv\Scripts\activate        # Windows
```

---

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

---

### 4️⃣ Install & run Ollama

```bash
ollama pull mistral
ollama run mistral
```

---

### 5️⃣ Build RAG database

```bash
python rag/ingest.py
```

---

### 6️⃣ Start backend

```bash
uvicorn backend.main:app --reload
```

---

### 7️⃣ Open frontend

Open in browser:

```
frontend/index.html
```

---

## 🧠 Key Features

### 🔹 Machine Learning

* Neural Network (TensorFlow)
* Random Forest
* XGBoost

### 🔹 Explainable AI

* SHAP-based feature importance
* Actionable vs non-actionable factors

### 🔹 RAG (Retrieval-Augmented Generation)

* ChromaDB vector database
* Sentence-transformer embeddings
* Clinical PDF corpus

### 🔹 LLM Integration

* Ollama + Mistral
* Generates:

  * Clinical explanation
  * Personalized advice

### 🔹 Frontend

* Interactive dashboard
* SHAP visualization
* Risk probability display
* Assessment history tracking

---

## 🧪 Example Output

* **Prediction:** Low Risk

* **Probability:** 24.6%

* **Top Features:**

  * Insulin
  * Glucose
  * BMI

* **Explanation:**
  Model-driven + clinically grounded reasoning

* **Advice:**
  Personalized lifestyle recommendations

---

## ⚠️ Disclaimer

This project is for **academic and educational purposes only**.
It does **not provide medical advice**. Always consult a qualified healthcare professional.

---

## 📌 Future Improvements

* Deploy on cloud (Render / AWS / GCP)
* Add authentication system
* Improve UI/UX animations
* Use stronger LLM (GPT / LLaMA 3)
* Add real-time patient data integration

---

## 👨‍💻 Author

**Aviral Gupta**
AI/ML Developer | Explainable AI Enthusiast

---

## ⭐ If you like this project

Give it a ⭐ on GitHub!

---
