
import os
from dotenv import load_dotenv

from src.rag_pipeline import create_rag_pipeline

# Load environment variables from .env
load_dotenv()

# Check Groq API key without displaying it
if not os.getenv("GROQ_API_KEY"):
    raise ValueError(
        "GROQ_API_KEY not found. Check your .env file."
    )

print("Initializing Finexa RAG Chatbot...")

ask_question = create_rag_pipeline()

print("\nFinexa RAG Chatbot is ready!")
print("Type 'exit' to stop.\n")

while True:

    question = input("You: ").strip()

    if question.lower() == "exit":
        print("Finexa chatbot stopped.")
        break

    if not question:
        continue

    try:

        answer = ask_question(question)

        print("\nFinexa:", answer)

    except Exception as error:

        print("\nError:", error)

    print("\n" + "=" * 60 + "\n")
