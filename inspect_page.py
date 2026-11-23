
from src.document_loader import load_documents

documents = load_documents()

# Page metadata is zero-indexed in PyPDFLoader.
page = documents[7]

print("\n========== PDF PAGE 8 ==========\n")
print(page.page_content)
