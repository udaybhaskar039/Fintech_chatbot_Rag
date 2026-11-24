


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
        output_dimensionality=384
    )

    return model
