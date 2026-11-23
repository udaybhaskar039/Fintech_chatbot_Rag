
from src.document_loader import load_documents
from src.rag_pipeline import split_policy_documents

pages = load_documents()

policies = split_policy_documents(pages)

print("\n========== EXTRACTED POLICIES ==========")

for doc in policies:
    print("\nPolicy ID:", doc.metadata.get("policy_id"))
    print("Title:", doc.metadata.get("policy_title"))
    print("Page:", doc.metadata.get("page"))
    print("Content:", doc.page_content[:500])
