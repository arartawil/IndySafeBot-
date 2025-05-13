import streamlit as st
from pathlib import Path
import sys

# Add backend to path
backend_path = Path(__file__).resolve().parents[1] / "backend"
sys.path.append(str(backend_path))

# Import the answer function
from rag_engine import retrieve_answer

# Page setup
st.set_page_config(page_title="IndySafeBot - QA", layout="centered")
st.title("🧠 IndySafeBot")
st.caption("Ask a public safety question and get an answer instantly.")

# Input field
question = st.text_input("❓ Enter your question:")

# Show answer when submitted
if question and st.button("🔍 Get Answer"):
    answer = retrieve_answer(question)
    st.markdown("**Answer:**")
    st.success(answer)
