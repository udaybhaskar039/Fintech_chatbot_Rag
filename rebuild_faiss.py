
from src.document_loader import load_documents
from src.rag_pipeline import split_policy_documents
from src.embedding_model import get_embedding_model
from src.vector_store import create_vector_store

print("Step 1: Loading PDF...")
pages = load_documents()

print("\nStep 2: Extracting policies...")
policies = split_policy_documents(pages)

print("\nTotal extracted policies:", len(policies))

print("\nStep 3: Loading embedding model...")
embedding_model = get_embedding_model()

print("\nStep 4: Creating new FAISS index...")
vector_store = create_vector_store(
    policies,
    embedding_model
)

print("\n========== REBUILD SUMMARY ==========")
print("Extracted policies:", len(policies))
print("FAISS vectors:", vector_store.index.ntotal)

print("\nFAISS rebuild completed!")
