
import os
import hashlib
import json

from dotenv import load_dotenv
from langchain_postgres import PGVector


load_dotenv()

COLLECTION_NAME = "finexa_policy_handbook_gemini_v2"
EMBEDDING_DIMENSION = 384


def get_database_url():

    database_url = os.getenv("DATABASE_URL")

    if not database_url:
        raise ValueError(
            "DATABASE_URL not found in .env file."
        )

    return database_url


def create_vector_store(documents, embedding_model):

    if not documents:
        raise ValueError(
            "No documents available to create vector store."
        )

    database_url = get_database_url()

    # Generate stable IDs to avoid duplicate document IDs
    ids = []

    for index, document in enumerate(documents):

        document_data = (
            str(index)
            + document.page_content
            + json.dumps(
                document.metadata or {},
                sort_keys=True,
                default=str
            )
        )

        document_id = hashlib.sha256(
            document_data.encode("utf-8")
        ).hexdigest()

        ids.append(document_id)

    vector_store = PGVector.from_documents(
        documents=documents,
        embedding=embedding_model,
        connection=database_url,
        collection_name=COLLECTION_NAME,
        embedding_length=EMBEDDING_DIMENSION,
        ids=ids,
        use_jsonb=True
    )

    print(
        "Supabase PGVector database created successfully."
    )

    print(
        "Total indexed documents:",
        len(documents)
    )

    return vector_store


def load_vector_store(embedding_model):

    database_url = get_database_url()

    vector_store = PGVector(
        embedding_model,
        connection=database_url,
        collection_name=COLLECTION_NAME,
        embedding_length=EMBEDDING_DIMENSION,
        use_jsonb=True
    )

    print(
        "Supabase PGVector collection loaded successfully."
    )

    return vector_store

