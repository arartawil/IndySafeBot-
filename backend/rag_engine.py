from sentence_transformers import SentenceTransformer
from transformers import pipeline
import faiss
import pickle
import time
from tqdm import tqdm

# Indicator: Load embedding model
print("🔍 Loading sentence transformer model...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Indicator: Load FLAN-T5 model
print("🤖 Loading FLAN-T5 model for text2text-generation...")
for _ in tqdm(range(3), desc="⌛ Loading text2text model..."):
    time.sleep(0.7)
qa_pipeline = pipeline("text2text-generation", model="google/flan-t5-base")

# ✅ Indicator: Load FAISS index + documents
print("📂 Loading FAISS index and documents...")
for _ in tqdm(range(3), desc="📦 Initializing vector store..."):
    time.sleep(0.5)
with open("vector_store/faiss_index.pkl", "rb") as f:
    faiss_index, documents = pickle.load(f)

def truncate_context(text, max_chars=1500):
    return text[:max_chars] + "..." if len(text) > max_chars else text

def retrieve_answer(query):
    print("\n📝 Embedding query...")
    query_vec = embedder.encode([query])

    print("🔍 Searching for relevant context...")
    _, idx = faiss_index.search(query_vec, k=3)

    print("📚 Assembling context...")
    raw_context = "\n".join([documents[i] for i in idx[0]])
    context = truncate_context(raw_context)

    print("💡 Generating answer...")
    prompt = f"Answer the question using the following context:\n{context}\n\nQuestion: {query}"
    result = qa_pipeline(prompt, max_length=256, do_sample=False)

    print("✅ Answer ready!\n")
    return result[0]['generated_text']