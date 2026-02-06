"""
FastAPI wrapper for the Local RAG System.
Provides REST API endpoints for querying documents and chat functionality.
"""
import sys
from pathlib import Path

# Add the parent directory to sys.path to import from src
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List
import logging

# Import from src modules
from src.constants import (
    EMBEDDING_MODEL_PATH,
    EMBEDDING_DIMENSION,
    OLLAMA_HOST,
    OLLAMA_PORT,
    OLLAMA_MODEL_NAME,
    OPENSEARCH_HOST,
    OPENSEARCH_PORT,
    OPENSEARCH_INDEX,
)

from src.chat import generate_response_streaming
from src.embeddings import get_embedding_model
from src.opensearch_client import OpenSearchClient, hybrid_search  # Import both class and function

from src.utils import setup_logging

# Initialize logger
setup_logging()
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Local RAG System API",
    description="A privacy-friendly RAG system for querying personal documents locally",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for models (lazy loading)
_embedding_model = None
_opensearch_client = None


def get_embedding_model_instance():
    """Lazy load the embedding model."""
    global _embedding_model
    if _embedding_model is None:
        logger.info(f"Loading embedding model from {EMBEDDING_MODEL_PATH}")
        _embedding_model = get_embedding_model()
    return _embedding_model


def get_opensearch_client_instance() -> OpenSearchClient:
    """Lazy load the OpenSearch client."""
    global _opensearch_client
    if _opensearch_client is None:
        logger.info(f"Connecting to OpenSearch at {OPENSEARCH_HOST}:{OPENSEARCH_PORT}")
        _opensearch_client = OpenSearchClient(
            host=OPENSEARCH_HOST,
            port=OPENSEARCH_PORT,
            index_name=OPENSEARCH_INDEX,
        )
    return _opensearch_client


# Pydantic models for request/response
class QueryRequest(BaseModel):
    """Request model for RAG query."""
    query: str = Field(..., description="The user's question or query")
    top_k: int = Field(default=3, ge=1, le=10, description="Number of top results to retrieve")
    use_rag: bool = Field(default=True, description="Whether to use RAG for context retrieval")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="LLM temperature setting")
    chat_history: Optional[List[dict]] = Field(default=None, description="Previous chat history")


class QueryResponse(BaseModel):
    """Response model for RAG query."""
    answer: str = Field(..., description="The generated answer")
    sources: Optional[List[dict]] = Field(default=None, description="Source documents used")
    query: str = Field(..., description="The original query")


class SearchRequest(BaseModel):
    """Request model for document search."""
    query: str = Field(..., description="Search query")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of results to return")


class SearchResult(BaseModel):
    """Model for a single search result."""
    text: str
    score: float
    metadata: Optional[dict] = None


class SearchResponse(BaseModel):
    """Response model for document search."""
    results: List[SearchResult]
    total: int


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    opensearch: str
    embedding_model: str
    llm: str


# API Endpoints
@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Welcome to the Local RAG System API",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Check the health status of all services."""
    status = {
        "status": "healthy",
        "opensearch": "unknown",
        "embedding_model": "unknown",
        "llm": "unknown",
    }
    
    # Check OpenSearch
    try:
        client = get_opensearch_client_instance()
        if client.ping():
            status["opensearch"] = "connected"
        else:
            status["opensearch"] = "disconnected"
            status["status"] = "degraded"
    except Exception as e:
        status["opensearch"] = f"error: {str(e)}"
        status["status"] = "degraded"
    
    # Check embedding model
    try:
        model = get_embedding_model_instance()
        if model is not None:
            status["embedding_model"] = "loaded"
        else:
            status["embedding_model"] = "not loaded"
            status["status"] = "degraded"
    except Exception as e:
        status["embedding_model"] = f"error: {str(e)}"
        status["status"] = "degraded"
    
    # Check LLM (Ollama)
    try:
        import requests
        response = requests.get(f"http://{OLLAMA_HOST}:{OLLAMA_PORT}/api/tags", timeout=5)
        if response.status_code == 200:
            status["llm"] = f"connected ({OLLAMA_MODEL_NAME})"
        else:
            status["llm"] = "disconnected"
            status["status"] = "degraded"
    except Exception as e:
        status["llm"] = f"error: {str(e)}"
        status["status"] = "degraded"
    
    return status


@app.post("/query", response_model=QueryResponse, tags=["RAG"])
async def query_documents(request: QueryRequest):
    """
    Query the RAG system with a question.
    
    This endpoint retrieves relevant documents from OpenSearch and uses
    the LLM to generate a contextual response.
    """
    try:
        logger.info(f"Processing query: {request.query}")
        
        # Get streaming response using the chat module
        stream = generate_response_streaming(
            query=request.query,
            use_hybrid_search=request.use_rag,
            num_results=request.top_k,
            temperature=request.temperature,
            chat_history=request.chat_history or [],
        )
        
        # Collect the streamed response
        answer = ""
        if stream:
            for chunk in stream:
                if hasattr(chunk, 'message') and hasattr(chunk.message, 'content'):
                    answer += chunk.message.content
                elif isinstance(chunk, dict) and 'message' in chunk:
                    answer += chunk['message'].get('content', '')
        
        return QueryResponse(
            answer=answer,
            sources=None,
            query=request.query,
        )
    
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search", response_model=SearchResponse, tags=["Search"])
async def search_documents(request: SearchRequest):
    """
    Search for documents without generating an LLM response.
    
    Returns the raw search results from the hybrid search.
    """
    try:
        logger.info(f"Searching for: {request.query}")
        
        # Get embedding model and generate query embedding
        model = get_embedding_model_instance()
        query_embedding = model.encode(request.query).tolist()
        
        # Perform hybrid search
        results = hybrid_search(
            query=request.query,
            query_embedding=query_embedding,
            top_k=request.top_k,
        )
        
        # Format results
        search_results = [
            SearchResult(
                text=hit.get("_source", {}).get("text", ""),
                score=hit.get("_score", 0.0),
                metadata=hit.get("_source", {}).get("metadata", {}),
            )
            for hit in results if isinstance(hit, dict)
        ]
        
        return SearchResponse(
            results=search_results,
            total=len(search_results),
        )
    
    except Exception as e:
        logger.error(f"Error searching documents: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/embed", tags=["Utilities"])
async def generate_embedding_endpoint(text: str):
    """
    Generate embeddings for a given text.
    
    Useful for testing and debugging the embedding model.
    """
    try:
        model = get_embedding_model_instance()
        embedding = model.encode(text).tolist()
        return {
            "text": text,
            "embedding_dimension": len(embedding),
            "embedding": embedding[:10],  # Return only first 10 values for brevity
        }
    except Exception as e:
        logger.error(f"Error generating embedding: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize models and connections on startup."""
    logger.info("Starting Local RAG System API...")
    logger.info(f"Embedding model: {EMBEDDING_MODEL_PATH}")
    logger.info(f"LLM model: {OLLAMA_MODEL_NAME}")
    logger.info(f"OpenSearch: {OPENSEARCH_HOST}:{OPENSEARCH_PORT}")


# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down Local RAG System API...")