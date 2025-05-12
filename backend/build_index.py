import os
import pickle
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# === Step 1: Load and Chunk Text ===
def load_text_chunks(folder_path):
    chunks = []
    print(f"🔍 Loading text files from: {folder_path}")
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            print(f"📄 Reading file: {filename}")
            with open(os.path.join(folder_path, filename), "r", encoding="utf-8") as f:
                text = f.read()
                # Chunk by paragraphs (each ~50+ characters)
                paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 50]
                chunks.extend(paragraphs)
    return chunks

# === Step 2: Embed the Text Chunks ===
def embed_chunks(chunks, model_name="all-MiniLM-L6-v2"):
    print(f"🔗 Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)
    embeddings = []
    print(f"🔄 Embedding {len(chunks)} text chunks...")
    for chunk in tqdm(chunks):
        emb = model.encode(chunk)
        embeddings.append(emb)
    return embeddings

# === Step 3: Create and Save FAISS Index ===
def save_faiss_index(embeddings, chunks, output_path="vector_store/faiss_index.pkl"):
    print("💾 Creating and saving FAISS index...")
    dim = len(embeddings[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump((index, chunks), f)

    print(f"✅ Saved FAISS index with {len(chunks)} chunks to {output_path}")

# === Main Execution ===
if __name__ == "__main__":
    import numpy as np

    folder_path = "knowledge_base"
    chunks = load_text_chunks(folder_path)
    embeddings = embed_chunks(chunks)
    save_faiss_index(embeddings, chunks)
