
import re
import json
import uuid

from langchain_groq import ChatGroq
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_postgres import PGEngine, PGVectorStore

from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url

from src.intent_classifier import classify_intent
from src.document_loader import load_documents
from src.embedding_model import get_embedding_model
from src.vector_store import get_database_url


# =========================================
# 1. RETRIEVAL SETTINGS
# =========================================

POLICY_TOP_K = 10
OPERATIONS_TOP_K = 8
FINAL_CONTEXT_DOCS = 4

POLICY_DISTANCE_THRESHOLD = 2.5
OPERATIONS_DISTANCE_THRESHOLD = 2.5

# Dedicated Gemini database storage
SCHEMA_NAME = "finexa_gemini"
TABLE_NAME = "finexa_policy_embeddings"
EMBEDDING_DIMENSION = 3072


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

def extract_policy_id(content):

    match = POLICY_ID_PATTERN.search(content)

    if match:
        return match.group(0).upper()

    return None


# =========================================
# 4. POLICY DOCUMENT CHUNKING
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
                    "WARNING: Policy ID not found:",
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
                    "Skipping metadata-only section:",
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


def tokenize(content):

    words = re.findall(
        r"\b[a-z0-9]+\b",
        content.lower()
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

    # POS matching

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

        if "kyc" in content or "video-kyc" in content:
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
# 7. DOCUMENT RETRIEVAL
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

        if distance <= threshold and keyword_score > 0:

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

        print("\nFinal Rank:", rank)
        print("Policy ID:", doc.metadata.get("policy_id"))
        print("Policy Title:", doc.metadata.get("policy_title"))
        print("Distance:", round(item["distance"], 4))
        print("Keyword Score:", round(item["keyword_score"], 4))

    print("\nTotal PGVector Results:", len(results))
    print("Accepted Candidates:", len(candidates))
    print("Final Context Documents:", len(selected))

    return [
        item["document"]
        for item in selected
    ]


# =========================================
# 8. DATABASE CONNECTION URLS
# =========================================

def get_sync_database_url():

    database_url = make_url(get_database_url())

    # Use Psycopg 3 for synchronous SQLAlchemy
    if database_url.drivername in [
        "postgresql",
        "postgres",
        "postgresql+psycopg2",
        "postgresql+asyncpg"
    ]:

        database_url = database_url.set(
            drivername="postgresql+psycopg"
        )

    return database_url


def get_async_database_url():

    database_url = make_url(get_database_url())

    query_params = dict(database_url.query)

    # Convert to asyncpg
    if database_url.drivername in [
        "postgresql",
        "postgres",
        "postgresql+psycopg",
        "postgresql+psycopg2"
    ]:

        database_url = database_url.set(
            drivername="postgresql+asyncpg"
        )

    # Convert sslmode to asyncpg ssl parameter
    sslmode = query_params.pop("sslmode", None)

    if sslmode and "ssl" not in query_params:
        query_params["ssl"] = "require"

    database_url = database_url.set(
        query=query_params
    )

    return database_url.render_as_string(
        hide_password=False
    )


# =========================================
# 9. GENERATE STABLE DOCUMENT IDS
# =========================================

def generate_document_id(document):

    document_data = (
        document.page_content
        + json.dumps(
            document.metadata or {},
            sort_keys=True,
            default=str
        )
    )

    return str(
        uuid.uuid5(
            uuid.NAMESPACE_URL,
            document_data
        )
    )


# =========================================
# 10. INITIALIZE GEMINI PGVECTOR STORE
# =========================================

def initialize_gemini_vector_store(
    documents,
    embedding_model
):

    if not documents:

        raise ValueError(
            "No policy documents available for indexing."
        )

    print("\nConnecting to Supabase...")

    # -------------------------------------
    # A. Create/check dedicated schema
    # -------------------------------------

    sync_engine = create_engine(
        get_sync_database_url()
    )

    table_exists = False
    existing_ids = set()

    try:

        with sync_engine.begin() as connection:

            connection.execute(
                text(
                    f'CREATE SCHEMA IF NOT EXISTS "{SCHEMA_NAME}"'
                )
            )

            table_exists = connection.execute(
                text(
                    "SELECT to_regclass(:table_name)"
                ),
                {
                    "table_name":
                    f"{SCHEMA_NAME}.{TABLE_NAME}"
                }
            ).scalar() is not None

            if table_exists:

                dimension = connection.execute(
                    text("""
                        SELECT format_type(
                            atttypid,
                            atttypmod
                        )
                        FROM pg_attribute
                        WHERE attrelid = to_regclass(:table_name)
                        AND attname = 'embedding'
                        AND NOT attisdropped
                    """),
                    {
                        "table_name":
                        f"{SCHEMA_NAME}.{TABLE_NAME}"
                    }
                ).scalar()

                print(
                    "Existing Gemini embedding dimension:",
                    dimension
                )

                if dimension != f"vector({EMBEDDING_DIMENSION})":

                    raise ValueError(
                        "Incorrect Gemini vector dimension. "
                        f"Expected vector({EMBEDDING_DIMENSION}), "
                        f"got {dimension}"
                    )

                rows = connection.execute(
                    text(
                        f'SELECT langchain_id FROM '
                        f'"{SCHEMA_NAME}"."{TABLE_NAME}"'
                    )
                ).fetchall()

                existing_ids = {
                    str(row[0])
                    for row in rows
                }

                print(
                    "Existing indexed documents:",
                    len(existing_ids)
                )

    finally:

        sync_engine.dispose()

    # -------------------------------------
    # B. Create PGEngine
    # -------------------------------------

    print("\nInitializing Gemini PGEngine...")

    pg_engine = PGEngine.from_connection_string(
        url=get_async_database_url()
    )

    # -------------------------------------
    # C. Create separate vector table
    # -------------------------------------

    if not table_exists:

        print("\nCreating dedicated Gemini table...")

        pg_engine.init_vectorstore_table(
            table_name=TABLE_NAME,
            vector_size=EMBEDDING_DIMENSION,
            schema_name=SCHEMA_NAME,
            content_column="content",
            embedding_column="embedding",
            metadata_json_column="langchain_metadata",
            id_column="langchain_id",
            overwrite_existing=False
        )

        print(
            "Gemini vector table created successfully."
        )

    # -------------------------------------
    # D. Connect PGVectorStore
    # -------------------------------------

    vector_store = PGVectorStore.create_sync(
        engine=pg_engine,
        embedding_service=embedding_model,
        table_name=TABLE_NAME,
        schema_name=SCHEMA_NAME,
        content_column="content",
        embedding_column="embedding",
        metadata_json_column="langchain_metadata",
        id_column="langchain_id"
    )

    # -------------------------------------
    # E. Add only missing documents
    # -------------------------------------

    missing_documents = []
    missing_ids = []

    for document in documents:

        document_id = generate_document_id(document)

        if document_id not in existing_ids:

            missing_documents.append(document)
            missing_ids.append(document_id)

    if missing_documents:

        print(
            "\nNew documents to index:",
            len(missing_documents)
        )

        vector_store.add_documents(
            documents=missing_documents,
            ids=missing_ids
        )

        print(
            "New Gemini documents indexed successfully."
        )

    else:

        print(
            "\nAll policy documents are already indexed."
        )

    print(
        "Total documents expected:",
        len(documents)
    )

    print(
        "Previously indexed documents:",
        len(existing_ids)
    )

    print(
        "New documents indexed:",
        len(missing_documents)
    )

    return vector_store


# =========================================
# 11. CREATE RAG PIPELINE
# =========================================

def create_rag_pipeline():

    print("\nInitializing Finexa RAG...")

    # -------------------------------------
    # A. Load PDF documents
    # -------------------------------------

    documents = load_documents()

    if not documents:

        raise ValueError(
            "No PDF documents found in data/raw."
        )

    print(
        "Loaded policy PDF documents:",
        len(documents)
    )

    # -------------------------------------
    # B. Extract policy-specific documents
    # -------------------------------------

    policy_documents = split_policy_documents(
        documents
    )

    if not policy_documents:

        raise ValueError(
            "No policy documents extracted. "
            "Check PDF headings and policy IDs."
        )

    # -------------------------------------
    # C. Initialize Gemini embeddings
    # -------------------------------------

    print("\nLoading Gemini embedding model...")

    embedding_model = get_embedding_model()

    # -------------------------------------
    # D. Initialize Supabase PGVector
    # -------------------------------------

    print("\nConnecting to Gemini PGVector storage...")

    vector_store = initialize_gemini_vector_store(
        policy_documents,
        embedding_model
    )

    print(
        "\nGemini PGVector initialized successfully."
    )

    # -------------------------------------
    # E. Initialize Groq LLM
    # -------------------------------------

    print("\nInitializing Groq LLM...")

    llm = ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0
    )

    # -------------------------------------
    # F. Strict RAG prompt
    # -------------------------------------

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

        if not isinstance(question, str) or not question.strip():

            return "Please enter a valid question."

        question = question.strip()

        # ---------------------------------
        # A. Detect intent
        # ---------------------------------

        intent = classify_intent(question)

        print("\nDetected Intent:", intent)

        if intent == "greeting":

            return (
                "Hello! I'm Finexa, your Financial "
                "Operations Assistant. How can I help you?"
            )

        if intent == "out_of_domain":

            return (
                "I'm Finexa, a Financial Operations "
                "Assistant. I can help with financial "
                "policies and financial operations. "
                "Your question is outside my supported domain."
            )

        # ---------------------------------
        # B. Select retrieval settings
        # ---------------------------------

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

            return (
                "Unable to identify the financial intent."
            )

        # ---------------------------------
        # C. Retrieve relevant documents
        # ---------------------------------

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

        # ---------------------------------
        # D. Build context
        # ---------------------------------

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

        # ---------------------------------
        # E. Generate Groq response
        # ---------------------------------

        messages = prompt.format_messages(
            context=context,
            question=question
        )

        response = llm.invoke(messages)

        return response.content

    print("\nFinexa RAG initialized successfully.")

    return ask_question
