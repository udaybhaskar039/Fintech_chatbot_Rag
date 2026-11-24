
from pathlib import Path
from contextlib import asynccontextmanager
import os

from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


# =========================================
# 1. LOAD ENVIRONMENT VARIABLES
# =========================================

ROOT_DIR = Path(__file__).resolve().parent

load_dotenv(ROOT_DIR / ".env")


# =========================================
# 2. IMPORT RAG PIPELINE
# =========================================

from src.rag_pipeline import create_rag_pipeline


# =========================================
# 3. RAG PIPELINE STORAGE
# =========================================

rag_pipeline = None


# =========================================
# 4. FASTAPI LIFESPAN
# =========================================

@asynccontextmanager
async def lifespan(app: FastAPI):

    global rag_pipeline

    required_variables = [
        "GROQ_API_KEY",
        "GOOGLE_API_KEY",
        "DATABASE_URL"
    ]

    missing_variables = [
        variable
        for variable in required_variables
        if not os.getenv(variable)
    ]

    if missing_variables:

        raise RuntimeError(
            "Missing environment variables: "
            + ", ".join(missing_variables)
        )

    print("\n================================")
    print("Starting Finexa RAG API")
    print("================================")

    try:

        print("\nInitializing Finexa RAG pipeline...")

        rag_pipeline = create_rag_pipeline()

        print("\nFinexa RAG initialized successfully!")

    except Exception as error:

        print("\nRAG INITIALIZATION FAILED:")
        print(str(error))

        raise

    yield

    print("\nShutting down Finexa RAG API...")

    rag_pipeline = None


# =========================================
# 5. CREATE FASTAPI APPLICATION
# =========================================

app = FastAPI(
    title="Finexa RAG API",
    description="Financial Policy RAG Chatbot API",
    version="1.0.0",
    lifespan=lifespan
)


# =========================================
# 6. CORS CONFIGURATION
# =========================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"]
)


# =========================================
# 7. REQUEST MODEL
# =========================================

class ChatRequest(BaseModel):

    question: str


# =========================================
# 8. HEALTH CHECK
# =========================================

@app.get("/")
def root():

    return {
        "message": "Welcome to Finexa RAG API",
        "status": "running"
    }


@app.get("/health")
def health():

    return {
        "status": "running",
        "rag_initialized": rag_pipeline is not None
    }


# =========================================
# 9. CHAT ENDPOINT
# =========================================

@app.post("/chat")
def chat(request: ChatRequest):

    if rag_pipeline is None:

        raise HTTPException(
            status_code=503,
            detail="RAG pipeline is not initialized"
        )

    question = request.question.strip()

    if not question:

        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty"
        )

    try:

        print("\nReceived Question:", question)

        answer = rag_pipeline(question)

        print("\nGenerated Answer:", answer)

        return {
            "success": True,
            "question": question,
            "answer": answer
        }

    except Exception as error:

        print("\nRAG ERROR:", str(error))

        raise HTTPException(
            status_code=500,
            detail="Unable to generate an answer"
        )
