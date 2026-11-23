
from src.embedding_model import get_embedding_model
from src.vector_store import load_vector_store

print("Loading embedding model...")

embedding_model = get_embedding_model()

print("\nLoading existing FAISS index...")

vector_store = load_vector_store(embedding_model)

print("\n========== FAISS INDEX DETAILS ==========")

print("Total vectors:", vector_store.index.ntotal)

print("\n========== STORED POLICY DOCUMENTS ==========")

documents = vector_store.docstore._dict

for i, doc in enumerate(documents.values(), start=1):
    print(
        i,
        "| Policy ID:", doc.metadata.get("policy_id"),
        "| Title:", doc.metadata.get("policy_title")
    )

print("\nFAISS loading test completed.")
