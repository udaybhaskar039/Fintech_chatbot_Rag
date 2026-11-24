
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings


def get_embedding_model():

    api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key:
        raise ValueError(
            "GOOGLE_API_KEY not found in environment variables."
        )

    model = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001",
        google_api_key=api_key,
        output_dimensionality=768
    )

    # Verify actual embedding dimensions
    test_embedding = model.embed_query(
        "Finexa financial policy test"
    )

    actual_dimension = len(test_embedding)

    print(
        "ACTUAL EMBEDDING DIMENSION:",
        actual_dimension
    )

    if actual_dimension != 768:
        raise ValueError(
            f"Embedding dimension mismatch! "
            f"Expected 768, got {actual_dimension}"
        )

    print("Gemini embedding model initialized successfully.")

    return model
