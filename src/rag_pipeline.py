
# =========================================
# 9. GENERATE STABLE DOCUMENT IDS
# =========================================

def get_document_fingerprint(content, metadata):
    """
    Generate a consistent fingerprint using document content
    and metadata. This works even if an older UUID format
    was used during previous indexing.
    """

    if isinstance(metadata, str):
        try:
            metadata = json.loads(metadata)
        except json.JSONDecodeError:
            metadata = {}

    metadata = metadata or {}

    return (
        content
        + json.dumps(
            metadata,
            sort_keys=True,
            ensure_ascii=False,
            default=str
        )
    )


def generate_document_id(document):

    document_data = get_document_fingerprint(
        document.page_content,
        document.metadata or {}
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
    existing_fingerprints = set()

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

                # Fetch existing content and metadata.
                # Do not depend only on langchain_id because
                # the UUID generation algorithm may have changed.

                rows = connection.execute(
                    text(
                        f'''
                        SELECT content, langchain_metadata
                        FROM "{SCHEMA_NAME}"."{TABLE_NAME}"
                        '''
                    )
                ).fetchall()

                for row in rows:

                    fingerprint = get_document_fingerprint(
                        row[0],
                        row[1]
                    )

                    existing_fingerprints.add(fingerprint)

                print(
                    "Existing indexed documents:",
                    len(existing_fingerprints)
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
    # E. Add only genuinely missing documents
    # -------------------------------------

    missing_documents = []
    missing_ids = []

    processed_fingerprints = set()

    for document in documents:

        fingerprint = get_document_fingerprint(
            document.page_content,
            document.metadata or {}
        )

        # Skip documents already stored in Supabase.
        if fingerprint in existing_fingerprints:
            print(
                "Skipping already indexed policy:",
                (document.metadata or {}).get("policy_id")
            )
            continue

        # Skip repeated documents in the current input.
        if fingerprint in processed_fingerprints:
            print(
                "Skipping duplicate input document:",
                (document.metadata or {}).get("policy_id")
            )
            continue

        processed_fingerprints.add(fingerprint)

        document_id = generate_document_id(document)

        missing_documents.append(document)
        missing_ids.append(document_id)

    # -------------------------------------
    # F. Index missing documents
    # -------------------------------------

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

    # -------------------------------------
    # G. Final indexing summary
    # -------------------------------------

    print("\n========== INDEXING SUMMARY ==========")

    print(
        "Total documents expected:",
        len(documents)
    )

    print(
        "Previously indexed documents:",
        len(existing_fingerprints)
    )

    print(
        "New documents indexed:",
        len(missing_documents)
    )

    print(
        "Duplicate documents skipped:",
        len(documents) - len(missing_documents)
    )

    print("\nGemini PGVector initialization completed.")

    return vector_store
