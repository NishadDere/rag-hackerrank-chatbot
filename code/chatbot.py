# file: chatbot.py

import time
import sys
from code.answer_with_provenance import answer_question

# settings and state
conversation_history = []
MODE = "strict"        # or "hybrid"
SHOW_CITATIONS = True
SHOW_PREVIEWS = False  # show chunk previews under the answer
TYPING_DELAY = 0.02    # per-char typing delay for simulation

MAX_CONTEXT_TURNS = 4

def build_conversational_question(user_input):
    recent = conversation_history[-MAX_CONTEXT_TURNS:]
    history_text = ""
    for turn in recent:
        history_text += f"User: {turn['question']}\nBot: {turn['answer']}\n\n"
    return history_text + f"User: {user_input}\n"

def simulate_typing(text):
    """Simple console typing animation (ChatGPT style)."""
    for ch in text:
        sys.stdout.write(ch)
        sys.stdout.flush()
        time.sleep(TYPING_DELAY)
    print()  # final newline

def print_chunk_previews(previews):
    print("\n--- Retrieved chunk previews ---")
    for idx, src, p in previews:
        print(f"[Chunk {idx}] Source: {src}")
        print(f"  {p}\n")
    print("-------------------------------\n")

def chat():
    global MODE, SHOW_CITATIONS, SHOW_PREVIEWS
    print("\n--- RAG Chatbot Started ---")
    print("Type 'quit' to exit.")
    print("Commands: /mode strict|hybrid, /citations on|off, /preview on|off\n")

    while True:
        user_input = input("You: ").strip()

        if not user_input:
            continue

        # Built-in commands
        if user_input.lower() in ["quit", "exit", "bye"]:
            print("Bot: Goodbye!")
            break

        if user_input.startswith("/mode"):
            parts = user_input.split()
            if len(parts) >= 2 and parts[1] in ("strict", "hybrid"):
                MODE = parts[1]
                print(f"Mode set to: {MODE}")
            else:
                print("Usage: /mode strict|hybrid")
            continue

        if user_input.startswith("/citations"):
            parts = user_input.split()
            if len(parts) >= 2 and parts[1] in ("on", "off"):
                SHOW_CITATIONS = (parts[1] == "on")
                print(f"Citations: {'ON' if SHOW_CITATIONS else 'OFF'}")
            else:
                print("Usage: /citations on|off")
            continue

        if user_input.startswith("/preview"):
            parts = user_input.split()
            if len(parts) >= 2 and parts[1] in ("on", "off"):
                SHOW_PREVIEWS = (parts[1] == "on")
                print(f"Chunk previews: {'ON' if SHOW_PREVIEWS else 'OFF'}")
            else:
                print("Usage: /preview on|off")
            continue

        # normal Q/A
        contextual_q = build_conversational_question(user_input)

        # call pipeline
        result = answer_question(contextual_q, chat_history=conversation_history, mode=MODE, show_citations=SHOW_CITATIONS)

        # typing animation for answer
        answer_text = result["answer"]
        confidence = result["confidence"]
        previews = result.get("previews", [])

        # Simulate typing and then show confidence & previews
        print("\nBot: ", end="")
        simulate_typing(answer_text)

        # show confidence
        print(f"\nConfidence: {confidence*100:.0f}%")

        # optionally show chunk previews
        if SHOW_PREVIEWS and previews:
            print_chunk_previews(previews)

        # Save to history
        conversation_history.append({
            "question": user_input,
            "answer": answer_text
        })

        print("\n" + "-"*60 + "\n")

if __name__ == "__main__":
    chat()
