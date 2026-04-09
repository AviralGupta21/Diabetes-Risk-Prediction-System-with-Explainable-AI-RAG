import sys
import os
import requests
import time

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from model.explain import explain_instance
from rag.retriever import get_explanation_context, get_advice_context

OLLAMA_URL   = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "mistral"

DISCLAIMER = (
    "\n\n Disclaimer: This is not medical advice. "
    "This application is an academic demonstration only. "
    "Please consult a qualified physician for any health decisions."
)

def _call_ollama(prompt: str, retries: int = 2) -> str:
    for attempt in range(retries + 1):
        try:
            response = requests.post(
                OLLAMA_URL,
                json={
                    "model" : OLLAMA_MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.3,
                        "num_predict": 200,
                        "num_ctx"    : 2048
                    }
                },
                timeout=180
            )
            response.raise_for_status()
            return response.json().get("response", "").strip()

        except requests.exceptions.HTTPError as e:
            if attempt < retries:
                print(f"[pipeline] Ollama HTTP error (attempt {attempt+1}), retrying in 3s...")
                time.sleep(3)
            else:
                return f"[LLM Error] {str(e)}"

        except requests.exceptions.ConnectionError:
            return (
                "[LLM Unavailable] Ollama is not running. "
                "Please start it with: ollama serve"
            )
        except Exception as e:
            return f"[LLM Error] {str(e)}"

def _build_explanation_prompt(
    prediction: int,
    probability: float,
    confidence: float,
    top_features_rag: list,
    context_chunks: list
) -> str:
    prediction_label = "HIGH RISK for diabetes" if prediction == 1 else "LOW RISK for diabetes"

    features_text = "\n".join([
        f"  - {f}: SHAP value = {v:.4f} "
        f"({'increases' if v > 0 else 'decreases'} diabetes risk)"
        for f, v in top_features_rag
    ])

    context_text = "\n\n---\n\n".join(chunk[:300] for chunk in context_chunks) if context_chunks else "No context available."

    prompt = f"""You are a clinical AI assistant helping explain a diabetes risk prediction to a patient.

PREDICTION RESULT:
- Outcome: {prediction_label}
- Probability: {probability:.1%}
- Model Confidence: {confidence:.1f}%

TOP FACTORS INFLUENCING THIS PREDICTION (from SHAP analysis):
{features_text}

CLINICAL REFERENCE CONTEXT (from medical literature):
{context_text}

INSTRUCTIONS:
Write a clear, empathetic explanation (3-4 sentences) that:
1. States the prediction result plainly
2. Explains which factors most influenced it and why that makes clinical sense
3. Uses the clinical context above to ground your explanation in evidence
4. Uses simple language suitable for a non-medical audience
5. Does NOT recommend treatments or medications

Explanation:"""

    return prompt


def _build_advice_prompt(
    prediction: int,
    probability: float,
    actionable_names: list,
    non_actionable_names: list,
    user_input: dict,
    context_chunks: list
) -> str:
    prediction_label = "high risk" if prediction == 1 else "low risk"

    actionable_values = "\n".join([
        f"  - {f}: {user_input.get(f, 'N/A')}"
        for f in actionable_names
    ])

    non_actionable_text = (
        ", ".join(non_actionable_names)
        if non_actionable_names
        else "None"
    )

    context_text = "\n\n---\n\n".join(context_chunks) if context_chunks else "No context available."

    prompt = f"""You are a clinical AI assistant providing personalised lifestyle guidance based on a diabetes risk assessment.

PATIENT PROFILE:
- Risk Level: {prediction_label} ({probability:.1%} probability)
- Key Modifiable Risk Factors and Values:
{actionable_values}
- Non-Modifiable Factors (acknowledged but not addressed): {non_actionable_text}

CLINICAL GUIDELINES CONTEXT (from medical literature):
{context_text}

INSTRUCTIONS:
Write exactly 3-4 sentences of personalised lifestyle advice. 
Do NOT use numbered lists, bullet points, or headings.
Write in plain flowing prose only.
Address the modifiable risk factors above using the patient's actual values.
Ground recommendations in the clinical guidelines provided.
Do NOT prescribe medications. Use warm, encouraging language.

Personalised Advice:"""

    return prompt

def run_pipeline(user_input: dict) -> dict:
    print("[pipeline] Step 1: Running explain_instance...")
    explain_result = explain_instance(user_input)

    prediction   = explain_result["prediction"]
    probability  = explain_result["probability"]
    confidence   = explain_result["confidence"]
    top_features = explain_result["top_features"]         
    top_features_rag = explain_result["top_features_rag"] 
    actionable_features     = explain_result["actionable_features"]
    non_actionable_features = explain_result["non_actionable_features"]
    all_features = explain_result["all_features"]

    print(f"[pipeline] Prediction: {'High Risk' if prediction == 1 else 'Low Risk'} ({probability:.1%})")
    print(f"[pipeline] Top RAG features: {[f for f, v in top_features_rag]}")

    print("[pipeline] Step 2: Retrieving explanation context...")
    exp_context = get_explanation_context(top_features_rag)

    print("[pipeline] Step 3: Retrieving advice context...")
    adv_context = get_advice_context(
        actionable_features,
        non_actionable_features,
        prediction
    )

    print("[pipeline] Step 4: Generating explanation narrative...")
    explanation_prompt = _build_explanation_prompt(
        prediction        = prediction,
        probability       = probability,
        confidence        = confidence,
        top_features_rag  = top_features_rag,
        context_chunks    = exp_context["chunks"]
    )
    explanation_text = _call_ollama(explanation_prompt)

    print("[pipeline] Step 5: Generating personalised advice...")
    advice_prompt = _build_advice_prompt(
        prediction           = prediction,
        probability          = probability,
        actionable_names     = adv_context["actionable_names"],
        non_actionable_names = adv_context["non_actionable_names"],
        user_input           = user_input,
        context_chunks       = adv_context["chunks"]
    )
    advice_text = _call_ollama(advice_prompt)

    print("[pipeline] Complete.")

    return {
        "prediction"              : prediction,
        "probability"             : round(probability, 4),
        "confidence"              : confidence,
        "top_features"            : [(f, round(float(v), 4)) for f, v in top_features],
        "all_features"            : [(f, round(float(v), 4)) for f, v in all_features],
        "explanation_text"        : explanation_text,
        "advice_text"             : advice_text,
        "explanation_sources"     : exp_context["sources"],
        "advice_sources"          : adv_context["sources"],
        "actionable_features"     : adv_context["actionable_names"],
        "non_actionable_features" : adv_context["non_actionable_names"]
    }

if __name__ == "__main__":
    print("\n" + "="*55)
    print("  PIPELINE TEST")
    print("="*55)

    sample_input = {
        "Pregnancies"             : 2,
        "Glucose"                 : 150,
        "BloodPressure"           : 80,
        "SkinThickness"           : 25,
        "Insulin"                 : 100,
        "BMI"                     : 30.0,
        "DiabetesPedigreeFunction": 0.5,
        "Age"                     : 35
    }

    print(f"\nInput: {sample_input}\n")
    result = run_pipeline(sample_input)

    print("\n--- RESULT ---")
    print(f"Prediction  : {'High Risk' if result['prediction'] == 1 else 'Low Risk'}")
    print(f"Probability : {result['probability']}")
    print(f"Confidence  : {result['confidence']}%")

    print(f"\nTop 5 Features:")
    for f, v in result["top_features"]:
        print(f"  {f}: {v}")

    print(f"\nExplanation Sources : {result['explanation_sources']}")
    print(f"Advice Sources      : {result['advice_sources']}")
    print(f"Actionable          : {result['actionable_features']}")
    print(f"Non-Actionable      : {result['non_actionable_features']}")

    print(f"\n--- EXPLANATION ---\n{result['explanation_text']}")
    print(f"\n--- ADVICE ---\n{result['advice_text']}")

    print("\n" + "="*55)
    print("  TEST COMPLETE")
    print("="*55 + "\n")