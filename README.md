
# 🤖 IndySafeBot

**IndySafeBot** is an AI-powered public safety assistant that answers emergency-related questions using a Retrieval-Augmented Generation (RAG) pipeline. It is designed to help communities access reliable safety information instantly and improve over time through user-submitted corrections.

---

## 🚀 Inspiration

We were inspired by the real need for accessible, trustworthy public safety information during emergencies. Many citizens don't know how to act in situations like floods, volcanic activity, or vandalism reporting—and often, official resources are fragmented or hard to find. We wanted to build a simple AI assistant that empowers people with fast, clear, and localized safety guidance.

---

## 🤖 What It Does

IndySafeBot allows users to ask safety-related questions and receive AI-generated answers based on a curated knowledge base. It supports topics such as:

- What to do during natural disasters (e.g., floods, volcanic eruptions)
- How to report vandalism or graffiti
- How to prepare emergency kits
- Public health risks after disasters

Users can also submit corrections to improve responses, which are saved and later reindexed into the system.

---

## 🛠 How We Built It

- **Frontend**: Streamlit
- **Backend**: Python, FAISS, SentenceTransformers
- **Model**: Google FLAN-T5 via Hugging Face Transformers
- **Storage**: Local `.txt` knowledge files + Pickle vector store
- **Feedback Loop**: Text-based correction system with on-demand retraining

---

## 🧱 Challenges We Ran Into

- Prompt tuning for clean and reliable answers from the language model
- Streamlit session state issues with form submissions
- Repetitive or low-quality knowledge chunks causing poor context retrieval
- Ensuring safe and meaningful learning from user-submitted corrections

---

## 🏆 Accomplishments That We're Proud Of

- End-to-end working AI safety assistant in under 48 hours
- Real-time answer generation and learning from community feedback
- Lightweight, fully local, and highly extensible design

---

## 📚 What We Learned

- RAG pipelines are powerful but context quality is key
- User feedback mechanisms turn static tools into evolving systems
- Streamlit is great for fast, interactive AI demos

---

## 🔮 What's Next for IndySafeBot

- Add multilingual support and location-specific datasets
- Integrate with mobile apps and city emergency platforms
- Enable voice-based input
- Pilot with local governments for real-world testing

---

## 🧰 Built With

- **Python** – Core logic and backend
- **Streamlit** – Interactive frontend
- **SentenceTransformers** – Text embeddings
- **FAISS** – Vector similarity search
- **Transformers (Hugging Face)** – FLAN-T5 for answer generation
- **NumPy, Pickle** – Data handling
- **Local `.txt` Files** – Emergency knowledge base

---

## 💻 Run the App

1. Clone the repo  
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the Streamlit app:

```bash
streamlit run frontend/app.py
```

4. (Optional) Run console mode:

```bash
python backend/main.py
```

---

## 📂 Folder Structure

```
indy-safebot-rag/
├── backend/
│   ├── rag_engine.py
│   ├── main.py
│   └── knowledge_base/
├── frontend/
│   └── app.py
├── data/
│   └── user_feedback.txt
├── vector_store/
│   └── faiss_index.pkl
└── requirements.txt
```
