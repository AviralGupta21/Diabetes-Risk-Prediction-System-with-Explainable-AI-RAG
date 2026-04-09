import os
import chromadb
from chromadb.utils import embedding_functions

BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
CHROMA_DB_DIR = os.path.join(BASE_DIR, "chroma_db")

EXPLANATION_COLLECTION = "explanation_collection"
ADVICE_COLLECTION      = "advice_collection"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

TOP_K = 4

FEATURE_DESCRIPTIONS = {
    "Pregnancies"              : "number of pregnancies and gestational diabetes risk",
    "Glucose"                  : "plasma glucose concentration and blood sugar levels",
    "BloodPressure"            : "diastolic blood pressure and hypertension in diabetes",
    "SkinThickness"            : "triceps skin fold thickness and body fat percentage",
    "Insulin"                  : "serum insulin levels and insulin resistance",
    "BMI"                      : "body mass index, obesity and diabetes risk",
    "DiabetesPedigreeFunction" : "diabetes pedigree function and hereditary genetic risk",
    "Age"                      : "age as a risk factor for type 2 diabetes onset",
}

ACTIONABLE_FEATURES = {
    "Glucose", "BMI", "Insulin",
    "BloodPressure", "SkinThickness", "Pregnancies"
}

NON_ACTIONABLE_FEATURES = {
    "DiabetesPedigreeFunction", "Age"
}

def _get_client_and_collections():
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )
    client = chromadb.PersistentClient(path=CHROMA_DB_DIR)

    explanation_col = client.get_collection(
        name=EXPLANATION_COLLECTION,
        embedding_function=embedding_fn
    )
    advice_col = client.get_collection(
        name=ADVICE_COLLECTION,
        embedding_function=embedding_fn
    )
    return explanation_col, advice_col

try:
    _explanation_collection, _advice_collection = _get_client_and_collections()
    print("[retriever] ChromaDB collections loaded successfully.")
except Exception as e:
    print(f"[retriever] ERROR loading ChromaDB: {e}")
    print("[retriever] Have you run ingest.py yet?")
    _explanation_collection = None
    _advice_collection      = None


def _extract_feature_names(feature_tuples: list) -> list:
    return [f for f, v in feature_tuples]

def _build_explanation_query(feature_names: list) -> str:
    descriptions = [
        FEATURE_DESCRIPTIONS.get(f, f)
        for f in feature_names
    ]
    return (
        "Clinical significance of "
        + ", ".join(descriptions)
        + " in diabetes diagnosis and risk assessment"
    )

def _build_advice_query(actionable_names: list, prediction: int) -> str:
    if not actionable_names:
        return "general diabetes prevention and awareness lifestyle recommendations"

    descriptions = [
        FEATURE_DESCRIPTIONS.get(f, f)
        for f in actionable_names
    ]
    risk_context = "diabetes management" if prediction == 1 else "diabetes prevention"

    return (
        f"Lifestyle recommendations and interventions for {risk_context} "
        f"related to {', '.join(descriptions)}"
    )

def get_explanation_context(
    top_features_rag: list,
    top_k: int = TOP_K
) -> dict:
    if _explanation_collection is None:
        return {"query": "", "chunks": [], "sources": []}

    feature_names = _extract_feature_names(top_features_rag)
    query = _build_explanation_query(feature_names)

    try:
        results = _explanation_collection.query(
            query_texts=[query],
            n_results=top_k,
            include=["documents", "metadatas"]
        )

        chunks  = results["documents"][0] if results["documents"] else []
        metas   = results["metadatas"][0] if results["metadatas"] else []
        sources = [m.get("source", "unknown") for m in metas]

        return {
            "query"  : query,
            "chunks" : chunks,
            "sources": sources
        }

    except Exception as e:
        print(f"[retriever] Explanation query failed: {e}")
        return {"query": query, "chunks": [], "sources": []}


def get_advice_context(
    actionable_features: list,
    non_actionable_features: list,
    prediction: int,
    top_k: int = TOP_K
) -> dict:
    if _advice_collection is None:
        return {
            "query": "", "chunks": [], "sources": [],
            "actionable_names": [], "non_actionable_names": []
        }

    actionable_names     = _extract_feature_names(actionable_features)
    non_actionable_names = _extract_feature_names(non_actionable_features)

    query = _build_advice_query(actionable_names, prediction)

    try:
        results = _advice_collection.query(
            query_texts=[query],
            n_results=top_k,
            include=["documents", "metadatas"]
        )

        chunks  = results["documents"][0] if results["documents"] else []
        metas   = results["metadatas"][0] if results["metadatas"] else []
        sources = [m.get("source", "unknown") for m in metas]

        return {
            "query"               : query,
            "chunks"              : chunks,
            "sources"             : sources,
            "actionable_names"    : actionable_names,
            "non_actionable_names": non_actionable_names
        }

    except Exception as e:
        print(f"[retriever] Advice query failed: {e}")
        return {
            "query"               : query,
            "chunks"              : [],
            "sources"             : [],
            "actionable_names"    : actionable_names,
            "non_actionable_names": non_actionable_names
        }

if __name__ == "__main__":
    print("\n" + "="*55)
    print("  RETRIEVER TEST")
    print("="*55)

    test_top_features_rag = [
        ("Glucose", 0.1740),
        ("Insulin", -0.1448),
        ("Age",      0.0559)
    ]
    test_actionable = [
        ("Glucose", 0.1740),
        ("Insulin", -0.1448)
    ]
    test_non_actionable = [
        ("Age", 0.0559)
    ]
    test_prediction = 0

    print(f"\nSimulated top_features_rag : {test_top_features_rag}")
    print(f"Simulated actionable       : {test_actionable}")
    print(f"Simulated non_actionable   : {test_non_actionable}")
    print(f"Simulated prediction       : {'High Risk' if test_prediction == 1 else 'Low Risk'}")

    print("\n--- Explanation Context ---")
    exp = get_explanation_context(test_top_features_rag)
    print(f"Query   : {exp['query']}")
    print(f"Chunks  : {len(exp['chunks'])} retrieved")
    print(f"Sources : {set(exp['sources'])}")
    if exp['chunks']:
        print(f"\nTop chunk preview:\n{exp['chunks'][0][:300]}...")

    print("\n--- Advice Context ---")
    adv = get_advice_context(test_actionable, test_non_actionable, test_prediction)
    print(f"Query            : {adv['query']}")
    print(f"Chunks           : {len(adv['chunks'])} retrieved")
    print(f"Sources          : {set(adv['sources'])}")
    print(f"Actionable names : {adv['actionable_names']}")
    print(f"Non-actionable   : {adv['non_actionable_names']}")
    if adv['chunks']:
        print(f"\nTop chunk preview:\n{adv['chunks'][0][:300]}...")

    print("\n" + "="*55)
    print("  TEST COMPLETE")
    print("="*55 + "\n")