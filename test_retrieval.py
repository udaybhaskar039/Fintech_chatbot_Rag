
from src.embedding_model import get_embedding_model
from src.vector_store import load_vector_store

print("Loading embedding model...")
embedding_model = get_embedding_model()

print("\nLoading FAISS database...")
vector_store = load_vector_store(embedding_model)

questions = [
    "What happens if a customer is debited twice for one transaction?",
    "When will failed UPI money be refunded?",
    "What happens if the beneficiary bank is offline?",
    "What is the KYC verification process?",
    "How long does a merchant refund take?"
]

for question in questions:

    print("\n" + "=" * 70)
    print("QUESTION:", question)

    results = vector_store.similarity_search_with_score(
        question,
        k=3
    )

    print("\nRETRIEVED POLICIES:")

    for doc, score in results:

        print("\nPolicy ID:", doc.metadata.get("policy_id"))
        print("Title:", doc.metadata.get("policy_title"))
        print("Distance:", round(score, 4))
        print("Content:", doc.page_content[:300])
