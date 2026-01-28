"""
Run the RAG system as an API service.
Usage: python run_service.py
"""
import uvicorn
from src.constants import RAG_SERVICE_HOST, RAG_SERVICE_PORT

if __name__ == "__main__":
    uvicorn.run(
        "api.rag_api:app",
        host=RAG_SERVICE_HOST,
        port=RAG_SERVICE_PORT,
        reload=True
    )