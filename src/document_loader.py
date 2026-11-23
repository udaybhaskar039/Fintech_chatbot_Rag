
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader


def load_documents(folder_path="data/raw"):
    documents = []

    folder = Path(folder_path)
    pdf_files = list(folder.rglob("*.pdf"))

    if not pdf_files:
        print("No PDF documents found.")
        return documents

    for pdf_file in pdf_files:
        loader = PyPDFLoader(str(pdf_file))
        loaded_docs = loader.load()

        documents.extend(loaded_docs)

        print(f"Loaded: {pdf_file.name}")

    print(f"Total pages loaded: {len(documents)}")

    return documents
