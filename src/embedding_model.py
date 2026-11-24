

import os

from langchain_huggingface import HuggingFaceEndpointEmbeddings


def get_embedding_model():

    hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

    if not hf_token:
        raise ValueError(
            "HUGGINGFACEHUB_API_TOKEN not found."
        )

    model = HuggingFaceEndpointEmbeddings(
        model="sentence-transformers/all-MiniLM-L6-v2",
        task="feature-extraction",
        huggingfacehub_api_token=hf_token
    )

    return model

