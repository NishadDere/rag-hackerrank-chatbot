# file: chatbot.py

from code.answer_with_provenance import answer_question
from code.retriever_chroma import retrieve_chunks

conversation_history = []  # stores previous Q&A


def build_conversational_question(user_input):
    """
    Combine the conversation history with the user's new question.
    This helps maintain context across multiple turns.
    """

    history_text = ""
    for turn in conversation_history:
        history_text += f"User: {turn['question']}\nBot: {turn['answer']}\n\n"

    # The new question includes previous context
    full_question = history_text + f"User: {user_input}\n"

    return full_question


def chat():
    print("\n--- RAG Chatbot Started ---")
    print("Type 'quit' to exit.\n")

    while True:
        user_input = input("You: ")

        if user_input.lower() in ["quit", "exit", "bye"]:
            print("Bot: Goodbye!")
            break

        # Build question with conversation memory
        conversational_question = build_conversational_question(user_input)

        # Get final answer using your RAG system
        bot_answer = answer_question(conversational_question)

        # Save to history
        conversation_history.append({
            "question": user_input,
            "answer": bot_answer
        })

        print("\nBot:", bot_answer)
        print("\n" + "-"*60 + "\n")


if __name__ == "__main__":
    chat()
