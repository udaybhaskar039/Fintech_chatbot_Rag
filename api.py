
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager
from pathlib import Path
from dotenv import load_dotenv
import os

from src.rag_pipeline import create_rag_pipeline


# Load environment variables
ROOT_DIR = Path(__file__).resolve().parent
load_dotenv(ROOT_DIR / ".env")


# Store RAG pipeline
rag_pipeline = None


# Initialize RAG pipeline when API starts
@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag_pipeline

    if not os.getenv("GROQ_API_KEY"):
        raise RuntimeError("GROQ_API_KEY not configured")

    print("Initializing Finexa RAG...")
    rag_pipeline = create_rag_pipeline()
    print("Finexa RAG initialized successfully!")

    yield


app = FastAPI(
    title="Finexa RAG API",
    description="Financial Policy RAG Chatbot API",
    version="1.0",
    lifespan=lifespan
)


# CORS - temporary development configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"]
)


# Request format
class ChatRequest(BaseModel):
    question: str


# Health check
@app.get("/health")
def health():
    return {
        "status": "running",
        "rag_initialized": rag_pipeline is not None
    }


# Chat endpoint
@app.post("/chat")
def chat(request: ChatRequest):

    if rag_pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="RAG pipeline is not initialized"
        )

    if not request.question.strip():
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty"
        )

    try:
        answer = rag_pipeline(request.question)

        return {
            "success": True,
            "answer": answer
        }

    except Exception as error:
        print("RAG Error:", error)

        raise HTTPException(
            status_code=500,
            detail="Unable to generate an answer"
        )
