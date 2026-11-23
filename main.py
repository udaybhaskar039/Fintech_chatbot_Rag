
import os
from dotenv import load_dotenv

from src.rag_pipeline import create_rag_pipeline


def main():

    # Load environment variables
    load_dotenv()

    if not os.getenv("GROQ_API_KEY"):
        raise ValueError("GROQ_API_KEY not found in .env file.")

    print("Initializing Finexa RAG Assistant...")

    # Create RAG pipeline
    ask_question = create_rag_pipeline()

    print("\nFinexa RAG Assistant is ready!")
    print("Type 'exit' to stop.\n")

    while True:

        question = input("You: ")

        if question.lower() == "exit":
            print("Thank you for using Finexa!")
            break

        try:
            answer = ask_question(question)
            print("\nFinexa:", answer, "\n")

        except Exception as error:
            print("Error:", error)


if __name__ == "__main__":
    main()
