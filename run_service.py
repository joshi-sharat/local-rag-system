"""
Run the RAG system as an API service.
Usage: python run_service.py
"""
import os
import sys

# Set PYTHONPATH environment variable - this persists to uvicorn subprocess
project_root = os.path.dirname(os.path.abspath(__file__))
os.environ["PYTHONPATH"] = project_root

# Also add to current process
sys.path.insert(0, project_root)

import uvicorn
from src.constants import RAG_SERVICE_HOST, RAG_SERVICE_PORT

if __name__ == "__main__":
    print(f"Project root: {project_root}")
    print(f"Starting server at http://{RAG_SERVICE_HOST}:{RAG_SERVICE_PORT}")
    
    uvicorn.run(
        "api.rag_api:app",
        host=RAG_SERVICE_HOST,
        port=RAG_SERVICE_PORT,
        reload=True,
    )