
import re

from langchain_groq import ChatGroq
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from sqlalchemy import create_engine, text

from src.intent_classifier import classify_intent
from src.document_loader import load_documents
from src.embedding_model import get_embedding_model
from src.vector_store import (
    create_vector_store,
    load_vector_store,
    COLLECTION_NAME,
    get_database_url
)


# =========================================
# 1. RETRIEVAL SETTINGS
# =========================================

POLICY_TOP_K = 10
OPERATIONS_TOP_K = 8

FINAL_CONTEXT_DOCS = 4

# Temporary retrieval thresholds.
# We will calibrate these after testing PGVector results.
POLICY_DISTANCE_THRESHOLD = 2.5
OPERATIONS_DISTANCE_THRESHOLD = 2.5


# =========================================
# 2. POLICY PATTERNS
# =========================================

POLICY_ID_PATTERN = re.compile(
    r"\bPOL\d{3}\b",
    re.IGNORECASE
)

POLICY_HEADING_PATTERN = re.compile(
    r"(?im)^\s*(\d{1,2})\.\s+([^\n]{2,180})\s*$"
)


# =========================================
# 3. EXTRACT POLICY ID
# =========================================

def extract_policy_id(text):

    match = POLICY_ID_PATTERN.search(text)

    if match:
        return match.group(0).upper()

    return None


# =========================================
# 4. POLICY-AWARE DOCUMENT CHUNKING
# =========================================

def split_policy_documents(documents):

    policy_documents = []

    for page_doc in documents:

        page_text = page_doc.page_content
        page_metadata = page_doc.metadata or {}

        matches = list(
            POLICY_HEADING_PATTERN.finditer(page_text)
        )

        if not matches:
            continue

        for index, match in enumerate(matches):

            start = match.start()

            if index + 1 < len(matches):
                end = matches[index + 1].start()
            else:
                end = len(page_text)

            section_text = page_text[start:end].strip()

            heading = match.group(2).strip()

            valid_heading = any(
                word in heading.lower()
                for word in [
                    "policy",
                    "protocol",
                    "reversal"
                ]
            )

            if not valid_heading:
                continue

            policy_id = extract_policy_id(section_text)

            if not policy_id:

                print(
                    "\nWARNING: Policy ID not found:",
                    heading
                )

                continue

            useful_markers = [
                "policy question",
                "policy content",
                "policy action",
                "purpose",
                "scope",
                "procedure",
                "within",
                "must",
                "should",
                "requires"
            ]

            if not any(
                marker in section_text.lower()
                for marker in useful_markers
            ):

                print(
                    "\nSkipping metadata-only section:",
                    policy_id
                )

                continue

            metadata = dict(page_metadata)

            metadata.update({
                "policy_id": policy_id,
                "policy_title": heading,
                "document_type": "financial_policy"
            })

            policy_documents.append(
                Document(
                    page_content=section_text,
                    metadata=metadata
                )
            )

            print(
                f"Created policy document: "
                f"{policy_id} - {heading}"
            )

    print(
        "\nTotal policy-specific documents:",
        len(policy_documents)
    )

    return policy_documents


# =========================================
# 5. KEYWORD MATCHING
# =========================================

STOPWORDS = {
    "what", "when", "where", "which", "who",
    "how", "does", "do", "is", "are", "the",
    "a", "an", "and", "or", "to", "of", "in",
    "on", "for", "with", "if", "should", "can",
    "could", "would", "will", "be", "it", "this",
    "that", "my", "your", "their", "customer",
    "happens", "much", "time", "within"
}


def tokenize(text):

    words = re.findall(
        r"\b[a-z0-9]+\b",
        text.lower()
    )

    return {
        word for word in words
        if word not in STOPWORDS and len(word) > 1
    }


# =========================================
# 6. KEYWORD RERANKING
# =========================================

def calculate_keyword_score(question, document):

    question_lower = question.lower()
    content = document.page_content.lower()

    query_terms = tokenize(question)
    document_terms = tokenize(content)

    if not query_terms:
        return 0

    matched_terms = query_terms.intersection(
        document_terms
    )

    score = len(matched_terms) / len(query_terms)

    # POS duplicate debit matching

    is_pos_question = bool(
        re.search(
            r"\bpos\b|point of sale|merchant debit",
            question_lower
        )
    )

    if is_pos_question:

        pos_phrases = [
            "point of sale",
            "pos",
            "merchant settlement",
            "merchant debit",
            "duplicate batch",
            "primary authorization"
        ]

        if any(
            phrase in content
            for phrase in pos_phrases
        ):
            score += 0.75
        else:
            score -= 0.50

    # Network retry matching

    if (
        "network retry" in question_lower
        or "retry loops" in question_lower
    ):

        if any(
            phrase in content
            for phrase in [
                "network retry",
                "network re-try",
                "dual switch",
                "secondary debit"
            ]
        ):
            score += 0.75

    # KYC matching

    if re.search(r"\bkyc\b", question_lower):

        if (
            "kyc" in content
            or "video-kyc" in content
        ):
            score += 0.75

    # Explicit policy ID matching

    requested_ids = re.findall(
        r"\bPOL\d{3}\b",
        question.upper()
    )

    policy_id = (
        document.metadata or {}
    ).get("policy_id")

    for requested_id in requested_ids:

        if policy_id == requested_id:
            score += 1.0

    return score


# =========================================
# 7. RETRIEVAL AND RERANKING
# =========================================

def retrieve_relevant_documents(
    vector_store,
    question,
    top_k,
    threshold
):

    results = vector_store.similarity_search_with_score(
        question,
        k=top_k
    )

    candidates = []

    print("\n========== INITIAL RETRIEVAL ==========")

    for rank, (doc, distance) in enumerate(
        results,
        start=1
    ):

        distance = float(distance)

        metadata = doc.metadata or {}

        policy_id = metadata.get(
            "policy_id",
            "UNKNOWN"
        )

        keyword_score = calculate_keyword_score(
            question,
            doc
        )

        print(f"\nCandidate Rank: {rank}")
        print("Policy ID:", policy_id)
        print("Distance:", round(distance, 4))
        print("Keyword Score:", round(keyword_score, 4))
        print("Content:", doc.page_content[:250])

        if (
            distance <= threshold
            and keyword_score > 0
        ):

            candidates.append({
                "document": doc,
                "distance": distance,
                "keyword_score": keyword_score
            })

            print("Status: ACCEPTED")

        else:

            print("Status: REJECTED")

    candidates.sort(
        key=lambda item: (
            -item["keyword_score"],
            item["distance"]
        )
    )

    selected = candidates[:FINAL_CONTEXT_DOCS]

    print("\n========== FINAL RERANKING ==========")

    for rank, item in enumerate(selected, start=1):

        doc = item["document"]

        print(f"\nFinal Rank: {rank}")

        print(
            "Policy ID:",
            doc.metadata.get("policy_id", "UNKNOWN")
        )

        print(
            "Policy Title:",
            doc.metadata.get("policy_title", "Unknown")
        )

        print(
            "Distance:",
            round(item["distance"], 4)
        )

        print(
            "Keyword Score:",
            round(item["keyword_score"], 4)
        )

    print("\nTotal PGVector Results:", len(results))
    print("Accepted Candidates:", len(candidates))
    print("Final Context Documents:", len(selected))

    return [
        item["document"]
        for item in selected
    ]


# =========================================
# 8. CREATE RAG PIPELINE
# =========================================

def create_rag_pipeline():

    # Load PDF pages

    documents = load_documents()

    if not documents:

        raise ValueError(
            "No PDF documents found in data/raw."
        )

    # Extract policy-specific documents

    policy_documents = split_policy_documents(
        documents
    )

    if not policy_documents:

        raise ValueError(
            "No policy documents extracted. "
            "Check PDF headings and policy IDs."
        )

    # Load embedding model

    print("\nLoading embedding model...")

    embedding_model = get_embedding_model()

    # =====================================
    # 9. LOAD OR CREATE SUPABASE PGVECTOR
    # =====================================

    print("\nConnecting to Supabase PGVector...")

    database_url = get_database_url()

    engine = create_engine(database_url)

    try:

        indexed_count = 0

        with engine.connect() as connection:

            collection_table_exists = connection.execute(
                text("""
                    SELECT to_regclass(
                        'public.langchain_pg_collection'
                    )
                """)
            ).scalar()

            if collection_table_exists:

                collection_id = connection.execute(
                    text("""
                        SELECT uuid
                        FROM public.langchain_pg_collection
                        WHERE name = :collection_name
                    """),
                    {"collection_name": COLLECTION_NAME}
                ).scalar()

                if collection_id:

                    indexed_count = connection.execute(
                        text("""
                            SELECT COUNT(*)
                            FROM public.langchain_pg_embedding
                            WHERE collection_id = :collection_id
                        """),
                        {"collection_id": collection_id}
                    ).scalar()

        if indexed_count > 0:

            print(
                f"\nExisting PGVector collection found."
                f"\nIndexed documents: {indexed_count}"
            )

            vector_store = load_vector_store(
                embedding_model
            )

        else:

            print("\nNo indexed documents found.")

            print(
                "Creating Supabase PGVector index..."
            )

            vector_store = create_vector_store(
                policy_documents,
                embedding_model
            )

            print(
                "\nSupabase indexing completed."
            )

    finally:

        engine.dispose()

    # =====================================
    # 10. INITIALIZE GROQ LLM
    # =====================================

    llm = ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0
    )

    # =====================================
    # 11. STRICT RAG PROMPT
    # =====================================

    prompt = ChatPromptTemplate.from_template("""
You are Finexa, a Financial Operations Assistant.

Answer using ONLY the supplied policy documents.

STRICT RULES:

1. Never invent policy IDs or policy names.
2. Use the POLICY ID from document metadata.
3. Never guess a policy ID based on similar wording.
4. Keep each policy's information separate.
5. Do not combine information from different policies.
6. Use only the actual policy content.
7. Ignore CSV instance counts and compliance metadata.
8. If the answer is unavailable, say:
   "I do not have enough information to answer that question."
9. Provide a concise and clear answer.
10. Include the relevant policy ID and policy name.
11. Include time limits only when explicitly present in the context.
12. Do not invent supplementary actions.
13. If the context does not support the requested answer,
    clearly state that the information is unavailable.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
""")

    # =====================================
    # 12. ASK QUESTION
    # =====================================

    def ask_question(question):

        intent = classify_intent(question)

        print("\nDetected Intent:", intent)

        # Greeting

        if intent == "greeting":

            return (
                "Hello! I'm Finexa, your Financial "
                "Operations Assistant. How can I help you?"
            )

        # Out of domain

        if intent == "out_of_domain":

            return (
                "I'm Finexa, a Financial Operations "
                "Assistant. I can help with financial "
                "policies and financial operations. "
                "Your question is outside my supported domain."
            )

        # Select retrieval settings

        if intent == "policy":

            print(
                "Routing to: Financial Policy Retrieval"
            )

            top_k = POLICY_TOP_K
            threshold = POLICY_DISTANCE_THRESHOLD

        elif intent == "financial_operations":

            print(
                "Routing to: Financial Operations Retrieval"
            )

            top_k = OPERATIONS_TOP_K
            threshold = OPERATIONS_DISTANCE_THRESHOLD

        else:

            return "Unable to identify the financial intent."

        # Retrieve documents

        relevant_docs = retrieve_relevant_documents(
            vector_store,
            question,
            top_k,
            threshold
        )

        if not relevant_docs:

            return (
                "I couldn't find sufficiently relevant "
                "information in the financial policy "
                "documents to answer your question."
            )

        # Build context

        context_parts = []

        for index, doc in enumerate(
            relevant_docs,
            start=1
        ):

            metadata = doc.metadata or {}

            policy_id = metadata.get(
                "policy_id",
                "UNKNOWN"
            )

            policy_title = metadata.get(
                "policy_title",
                "Unknown Policy"
            )

            page = metadata.get(
                "page",
                "Unknown"
            )

            context_parts.append(
                f"""
DOCUMENT {index}

POLICY ID: {policy_id}
POLICY TITLE: {policy_title}
PDF PAGE: {page}

POLICY CONTENT:
{doc.page_content}
"""
            )

        context = "\n\n".join(context_parts)

        # Generate answer

        messages = prompt.format_messages(
            context=context,
            question=question
        )

        response = llm.invoke(messages)

        return response.content

    return ask_question

