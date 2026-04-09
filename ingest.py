import os
import chromadb
from chromadb.utils import embedding_functions
from langchain_text_splitters import RecursiveCharacterTextSplitter
import fitz

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

EXPLANATION_CORPUS_DIR = os.path.join(BASE_DIR, "corpus", "explanation")
ADVICE_CORPUS_DIR      = os.path.join(BASE_DIR, "corpus", "advice")
CHROMA_DB_DIR          = os.path.join(BASE_DIR, "chroma_db")

EXPLANATION_COLLECTION = "explanation_collection"
ADVICE_COLLECTION      = "advice_collection"

CHUNK_SIZE    = 500
CHUNK_OVERLAP = 50

EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def extract_text_from_pdf(pdf_path: str) -> str:
    text = ""
    try:
        doc = fitz.open(pdf_path)
        for page in doc:
            page_text = page.get_text()
            if page_text:
                text += page_text + "\n"
        doc.close()
        print(f"  [OK] Extracted text from: {os.path.basename(pdf_path)}")
    except Exception as e:
        print(f"  [ERROR] Could not read {pdf_path}: {e}")
    return text.strip()


def chunk_text(text: str, source_filename: str) -> list[dict]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_text(text)

    chunks = [c.strip() for c in chunks if len(c.strip()) > 80]

    return [
        {"text": chunk, "source": source_filename}
        for chunk in chunks
    ]


def load_pdfs_from_folder(folder_path: str) -> list[dict]:
    all_chunks = []

    if not os.path.exists(folder_path):
        print(f"  [WARNING] Folder not found: {folder_path}")
        return all_chunks

    pdf_files = [f for f in os.listdir(folder_path) if f.endswith(".pdf")]

    if not pdf_files:
        print(f"  [WARNING] No PDFs found in: {folder_path}")
        return all_chunks

    for pdf_file in pdf_files:
        pdf_path = os.path.join(folder_path, pdf_file)
        text = extract_text_from_pdf(pdf_path)
        if text:
            chunks = chunk_text(text, source_filename=pdf_file)
            all_chunks.extend(chunks)
            print(f"  [OK] {pdf_file} → {len(chunks)} chunks")

    return all_chunks


def ingest_into_collection(
    collection,
    chunks: list[dict],
    collection_name: str
):
    if not chunks:
        print(f"  [SKIP] No chunks to ingest for: {collection_name}")
        return

    documents = [c["text"]   for c in chunks]
    metadatas = [{"source": c["source"]} for c in chunks]
    ids       = [f"{collection_name}_{i}" for i in range(len(chunks))]

    BATCH_SIZE = 100
    total = len(chunks)

    for start in range(0, total, BATCH_SIZE):
        end = min(start + BATCH_SIZE, total)
        collection.add(
            documents=documents[start:end],
            metadatas=metadatas[start:end],
            ids=ids[start:end]
        )

    print(f"  [OK] Ingested {total} chunks into '{collection_name}'")

def main():
    print("\n" + "="*55)
    print("  RAG INGESTION PIPELINE")
    print("="*55)

    print(f"\n[1/5] Initialising ChromaDB at: {CHROMA_DB_DIR}")
    client = chromadb.PersistentClient(path=CHROMA_DB_DIR)

    print(f"\n[2/5] Loading embedding model: {EMBEDDING_MODEL}")
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )

    print("\n[3/5] Setting up ChromaDB collections")

    for name in [EXPLANATION_COLLECTION, ADVICE_COLLECTION]:
        try:
            client.delete_collection(name=name)
            print(f"  [RESET] Deleted existing collection: '{name}'")
        except Exception:
            pass 

    explanation_col = client.create_collection(
        name=EXPLANATION_COLLECTION,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"}
    )

    advice_col = client.create_collection(
        name=ADVICE_COLLECTION,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"}
    )

    print(f"  [OK] Created '{EXPLANATION_COLLECTION}'")
    print(f"  [OK] Created '{ADVICE_COLLECTION}'")

    print("\n[4/5] Loading and chunking PDFs")

    print(f"\n  → Explanation Corpus: {EXPLANATION_CORPUS_DIR}")
    explanation_chunks = load_pdfs_from_folder(EXPLANATION_CORPUS_DIR)

    print(f"\n  → Advice Corpus: {ADVICE_CORPUS_DIR}")
    advice_chunks = load_pdfs_from_folder(ADVICE_CORPUS_DIR)

    print("\n[5/5] Ingesting chunks into ChromaDB")

    print(f"\n  → Ingesting into '{EXPLANATION_COLLECTION}':")
    ingest_into_collection(explanation_col, explanation_chunks, EXPLANATION_COLLECTION)

    print(f"\n  → Ingesting into '{ADVICE_COLLECTION}':")
    ingest_into_collection(advice_col, advice_chunks, ADVICE_COLLECTION)

    print("\n" + "="*55)
    print("  INGESTION COMPLETE")
    print("="*55)
    print(f"  Explanation chunks : {len(explanation_chunks)}")
    print(f"  Advice chunks      : {len(advice_chunks)}")
    print(f"  Total chunks       : {len(explanation_chunks) + len(advice_chunks)}")
    print(f"  ChromaDB saved at  : {CHROMA_DB_DIR}")
    print("\n  You can now run the backend. Do NOT re-run")
    print("  this script unless your corpus has changed.")
    print("="*55 + "\n")


if __name__ == "__main__":
    main()