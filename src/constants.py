
import os

####################################################################################################
EMBEDDING_MODEL_PATH = "embedding_model/sentence-transformer" # "sentence-transformers/all-mpnet-base-v2"  # OR Path of local eg. "embedding_model/"" or the name of SentenceTransformer model eg. "sentence-transformers/all-mpnet-base-v2" from Hugging Face
ASSYMETRIC_EMBEDDING = False  # Flag for asymmetric embedding
EMBEDDING_DIMENSION = 768  # Embedding model settings
TEXT_CHUNK_SIZE = 300  # Maximum number of characters in each text chunk for

OLLAMA_MODEL_NAME = (
    "gemma3"  # Name of the model used in Ollama for chat functionality
)
OLLAMA_HOST = "192.168.4.93:11434"  # Host address for the Ollama server
####################################################################################################
# Dont change the following settings
####################################################################################################

# Logging
LOG_FILE_PATH = "logs/app.log"  # File path for the application log file
# OpenSearch settings
OPENSEARCH_HOST = "localhost"  # Hostname for the OpenSearch instance
OPENSEARCH_PORT = 9200  # Port number for OpenSearch
OPENSEARCH_INDEX = "documents"  # Index name for storing documents in OpenSearch

# ADD these to constants.py:


# Service Configuration (NEW)
RAG_SERVICE_HOST = os.getenv("RAG_SERVICE_HOST", "localhost")
RAG_SERVICE_PORT = int(os.getenv("RAG_SERVICE_PORT", "8080"))

# LLM Provider Selection (NEW)
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")  # "ollama" or "anthropic"
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://192.168.4.107:11434")
OLLAMA_MODEL_NAME = os.getenv("OLLAMA_MODEL_NAME", "gemma3")

# EXISTING (make configurable via env vars):
EMBEDDING_MODEL_PATH = os.getenv("EMBEDDING_MODEL_PATH", "sentence-transformers/all-mpnet-base-v2")
EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "768"))
TEXT_CHUNK_SIZE = int(os.getenv("TEXT_CHUNK_SIZE", "300"))

# OpenSearch (make configurable)
OPENSEARCH_HOST = os.getenv("OPENSEARCH_HOST", "localhost")
OPENSEARCH_PORT = int(os.getenv("OPENSEARCH_PORT", "9200"))
OPENSEARCH_INDEX = os.getenv("OPENSEARCH_INDEX", "documents")