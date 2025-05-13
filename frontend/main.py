from rag_engine import retrieve_answer, rebuild_faiss_index
import os

user_corrections = []

def run_quiz_mode():
    print("🧠 IndySafeBot - Terminal QA Mode")
    print("Type 'exit' to quit.\n")

    while True:
        query = input("❓ You: ")
        if query.lower() == "exit":
            break

        answer = retrieve_answer(query)
        print("🤖 Bot:", answer)

        feedback = input("👍 Was this answer correct? (yes/no): ").strip().lower()
        if feedback == "no":
            correction = input("📝 Please provide the correct answer: ").strip()
            user_corrections.append(f"Q: {query}\nA: {correction}")
            print("✅ Correction saved.")

    if user_corrections:
        print("\n💾 Saving corrections and rebuilding knowledge base...")
        os.makedirs("data", exist_ok=True)
        with open("data/user_feedback.txt", "a", encoding="utf-8") as f:
            f.write("\n\n".join(user_corrections) + "\n")
        rebuild_faiss_index()
        print("✅ Knowledge base updated and reindexed.")
    else:
        print("📭 No corrections submitted.")

    print("👋 Goodbye!")

if __name__ == "__main__":
    run_quiz_mode()
