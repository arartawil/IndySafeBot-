from sentence_transformers import SentenceTransformer
import faiss
import pickle
import numpy as np
import os

def rebuild_faiss_index():
    chunks = []
    folder = "knowledge_base"
    for file in os.listdir(folder):
        if file.endswith(".txt"):
            with open(os.path.join(folder, file), "r", encoding="utf-8") as f:
                content = f.read().split("\n\n")
                chunks += [c.strip() for c in content if len(c.strip()) > 30]

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(chunks, show_progress_bar=True)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))

    with open("faiss_index.pkl", "wb") as f:
        pickle.dump((index, chunks), f)

    print("✅ Knowledge indexed.")

rebuild_faiss_index()
