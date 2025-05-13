from sentence_transformers import SentenceTransformer
from transformers import pipeline
import faiss
import pickle
import os
import numpy as np

# Load models and index once
embedder = SentenceTransformer("all-MiniLM-L6-v2")
qa_pipeline = pipeline("text2text-generation", model="google/flan-t5-base")

with open("faiss_index.pkl", "rb") as f:
    faiss_index, documents = pickle.load(f)

# Helpers
def truncate_context(text, max_chars=1200):
    return text[:max_chars] + "..." if len(text) > max_chars else text

def detect_question_type(query):
    q = query.lower()
    if "compare" in q or "difference" in q:
        return "compare"
    elif "explain" in q or "why" in q:
        return "explain"
    elif "how" in q:
        return "how"
    elif "what" in q:
        return "what"
    return "general"

def retrieve_answer(query):
    query_vec = embedder.encode(query)
    query_vec = np.array([query_vec]).astype("float32")  # ✅ Fix here

    _, idx = faiss_index.search(query_vec, k=3)
    context = "\n".join([documents[i] for i in idx[0]])
    context = truncate_context(context)

    q_type = detect_question_type(query)
    if q_type == "compare":
        prompt = f"Compare the following based on the context:\n{context}\n\nQuestion: {query}"
    elif q_type == "explain":
        prompt = f"Explain in detail using the context:\n{context}\n\nQuestion: {query}"
    else:
        prompt = f"Use the following context to answer:\n{context}\n\nQuestion: {query}"

    result = qa_pipeline(prompt, max_length=256, do_sample=False)
    return result[0]['generated_text']


# Correction feedback
def save_correction(question, correction):
    os.makedirs("data", exist_ok=True)
    with open("data/user_feedback.txt", "a", encoding="utf-8") as f:
        f.write(f"Q: {question}\nA: {correction}\n\n")

def rebuild_faiss_index():
    print("🔄 Rebuilding FAISS index...")
    all_chunks = []

    kb_folder = "knowledge_base"
    if not os.path.exists(kb_folder):
        os.makedirs(kb_folder)

    # Read and print each file
    for file in os.listdir(kb_folder):
        if file.endswith(".txt"):
            file_path = os.path.join(kb_folder, file)
            print(f"📄 Reading file: {file_path}")
            with open(file_path, "r", encoding="utf-8") as f:
                chunks = [chunk.strip() for chunk in f.read().split("\n\n") if len(chunk.strip()) > 30]
                all_chunks.extend(chunks)

    # Include feedback
    feedback_path = "data/user_feedback.txt"
    if os.path.exists(feedback_path):
        print(f"📄 Reading feedback file: {feedback_path}")
        with open(feedback_path, "r", encoding="utf-8") as f:
            feedback_chunks = [chunk.strip() for chunk in f.read().split("\n\n") if len(chunk.strip()) > 30]
            all_chunks.extend(feedback_chunks)

    vectors = embedder.encode(all_chunks, show_progress_bar=True)
    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(np.array(vectors))

    with open("faiss_index.pkl", "wb") as f:
        pickle.dump((index, all_chunks), f)

    print("✅ FAISS index rebuilt.")