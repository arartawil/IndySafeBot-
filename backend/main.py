from rag_engine import retrieve_answer

def run_console_chat():
    print("🧠 IndySafeBot RAG Chat - Console Mode")
    print("Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("👋 Goodbye!")
            break
        response = retrieve_answer(user_input)
        print("Bot:", response)

if __name__ == "__main__":
    run_console_chat()
