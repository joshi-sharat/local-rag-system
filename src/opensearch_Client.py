"""
OpenSearch client for the Local RAG System.
Handles document indexing and hybrid search (BM25 + semantic).
"""
from opensearchpy import OpenSearch
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class OpenSearchClient:
    def __init__(self, host: str, port: int, index_name: str):
        """
        Initialize the OpenSearch client.
        
        Args:
            host: OpenSearch host address
            port: OpenSearch port
            index_name: Name of the index to use
        """
        self.host = host
        self.port = port
        self.index_name = index_name
        
        self.client = OpenSearch(
            hosts=[{"host": host, "port": port}],
            http_compress=True,
            use_ssl=False,
            verify_certs=False,
            ssl_show_warn=False,
        )
        logger.info(f"Connected to OpenSearch at {host}:{port}")

    def create_index(self, embedding_dimension: int = 768) -> bool:
        """
        Create the index with mappings for hybrid search.
        
        Args:
            embedding_dimension: Dimension of the embedding vectors
            
        Returns:
            True if index was created, False if it already exists
        """
        if self.client.indices.exists(index=self.index_name):
            logger.info(f"Index '{self.index_name}' already exists")
            return False
        
        index_body = {
            "settings": {
                "index": {
                    "knn": True,
                    "number_of_shards": 1,
                    "number_of_replicas": 0,
                },
            },
            "mappings": {
                "properties": {
                    "text": {
                        "type": "text",
                        "analyzer": "standard",
                    },
                    "embedding": {
                        "type": "knn_vector",
                        "dimension": embedding_dimension,
                        "method": {
                            "name": "hnsw",
                            "space_type": "cosinesimil",
                            "engine": "nmslib",
                            "parameters": {
                                "ef_construction": 128,
                                "m": 16,
                            },
                        },
                    },
                    "metadata": {
                        "type": "object",
                        "enabled": True,
                    },
                    "doc_id": {
                        "type": "keyword",
                    },
                    "filename": {
                        "type": "keyword",
                    },
                    "page_number": {
                        "type": "integer",
                    },
                    "chunk_index": {
                        "type": "integer",
                    },
                    "created_at": {
                        "type": "date",
                    },
                },
            },
        }
        
        self.client.indices.create(index=self.index_name, body=index_body)
        logger.info(f"Created index '{self.index_name}'")
        return True

    def index_document(
        self,
        doc_id: str,
        text: str,
        embedding: List[float],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict:
        """
        Index a single document.
        
        Args:
            doc_id: Unique document identifier
            text: Document text content
            embedding: Vector embedding of the text
            metadata: Optional metadata dictionary
            
        Returns:
            OpenSearch response
        """
        document = {
            "doc_id": doc_id,
            "text": text,
            "embedding": embedding,
            "metadata": metadata or {},
        }
        
        response = self.client.index(
            index=self.index_name,
            id=doc_id,
            body=document,
            refresh=True,
        )
        logger.debug(f"Indexed document: {doc_id}")
        return response

    def bulk_index(self, documents: List[Dict[str, Any]]) -> Dict:
        """
        Bulk index multiple documents.
        
        Args:
            documents: List of documents with keys: doc_id, text, embedding, metadata
            
        Returns:
            Bulk indexing response
        """
        if not documents:
            return {"indexed": 0}
        
        actions = []
        for doc in documents:
            actions.append({
                "index": {
                    "_index": self.index_name,
                    "_id": doc["doc_id"],
                }
            })
            actions.append({
                "doc_id": doc["doc_id"],
                "text": doc["text"],
                "embedding": doc["embedding"],
                "metadata": doc.get("metadata", {}),
            })
        
        response = self.client.bulk(body=actions, refresh=True)
        logger.info(f"Bulk indexed {len(documents)} documents")
        return response

    def hybrid_search(
        self,
        query_text: str,
        query_embedding: List[float],
        top_k: int = 5,
    ) -> Dict:
        """
        Perform hybrid search combining BM25 and semantic search.
        
        Args:
            query_text: Text query for BM25 search
            query_embedding: Vector embedding for semantic search
            top_k: Number of results to return
            
        Returns:
            Search results
        """
        query = {
            "size": top_k,
            "query": {
                "hybrid": {
                    "queries": [
                        {
                            "match": {
                                "text": {
                                    "query": query_text,
                                }
                            }
                        },
                        {
                            "knn": {
                                "embedding": {
                                    "vector": query_embedding,
                                    "k": top_k,
                                }
                            }
                        },
                    ]
                }
            },
        }
        
        response = self.client.search(
            index=self.index_name,
            body=query,
            params={"search_pipeline": "nlp-search-pipeline"},
        )
        logger.debug(f"Hybrid search returned {len(response['hits']['hits'])} results")
        return response

    def semantic_search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
    ) -> Dict:
        """
        Perform semantic search using vector similarity.
        
        Args:
            query_embedding: Vector embedding of the query
            top_k: Number of results to return
            
        Returns:
            Search results
        """
        query = {
            "size": top_k,
            "query": {
                "knn": {
                    "embedding": {
                        "vector": query_embedding,
                        "k": top_k,
                    }
                }
            },
        }
        
        response = self.client.search(index=self.index_name, body=query)
        logger.debug(f"Semantic search returned {len(response['hits']['hits'])} results")
        return response

    def keyword_search(self, query_text: str, top_k: int = 5) -> Dict:
        """
        Perform keyword search using BM25.
        
        Args:
            query_text: Text query
            top_k: Number of results to return
            
        Returns:
            Search results
        """
        query = {
            "size": top_k,
            "query": {
                "match": {
                    "text": {
                        "query": query_text,
                    }
                }
            },
        }
        
        response = self.client.search(index=self.index_name, body=query)
        logger.debug(f"Keyword search returned {len(response['hits']['hits'])} results")
        return response

    def delete_document(self, doc_id: str) -> Dict:
        """
        Delete a document by ID.
        
        Args:
            doc_id: Document ID to delete
            
        Returns:
            Delete response
        """
        response = self.client.delete(
            index=self.index_name,
            id=doc_id,
            refresh=True,
        )
        logger.info(f"Deleted document: {doc_id}")
        return response

    def list_documents(self, limit: int = 100, offset: int = 0) -> Dict:
        """
        List documents in the index.
        
        Args:
            limit: Maximum number of documents to return
            offset: Number of documents to skip
            
        Returns:
            Search results with document list
        """
        query = {
            "size": limit,
            "from": offset,
            "query": {"match_all": {}},
            "_source": ["doc_id", "text", "metadata"],
        }
        
        response = self.client.search(index=self.index_name, body=query)
        return response

    def ping(self) -> bool:
        """Check if OpenSearch is reachable."""
        return self.client.ping()

    def get_document_count(self) -> int:
        """Get the total number of documents in the index."""
        try:
            response = self.client.count(index=self.index_name)
            return response.get("count", 0)
        except Exception:
            return 0


# Convenience function to create a client instance
def get_opensearch_client(
    host: str = "localhost",
    port: int = 9200,
    index_name: str = "rag_documents",
) -> OpenSearchClient:
    """
    Create and return an OpenSearch client instance.
    
    Args:
        host: OpenSearch host
        port: OpenSearch port
        index_name: Index name
        
    Returns:
        OpenSearchClient instance
    """
    return OpenSearchClient(host=host, port=port, index_name=index_name)


# =============================================================================
# STANDALONE FUNCTIONS (for backward compatibility with existing code)
# =============================================================================

# Global client instance for standalone functions
_default_client: Optional[OpenSearchClient] = None


def _get_default_client() -> OpenSearchClient:
    """Get or create the default client instance."""
    global _default_client
    if _default_client is None:
        # Import constants here to avoid circular imports
        try:
            from src.constants import OPENSEARCH_HOST, OPENSEARCH_PORT, INDEX_NAME
            _default_client = OpenSearchClient(
                host=OPENSEARCH_HOST,
                port=OPENSEARCH_PORT,
                index_name=INDEX_NAME,
            )
        except ImportError:
            # Fallback to defaults
            _default_client = OpenSearchClient(
                host="localhost",
                port=9200,
                index_name="rag_documents",
            )
    return _default_client


def hybrid_search(
    query_text: str,
    query_embedding: List[float],
    top_k: int = 5,
    client: Optional[OpenSearchClient] = None,
) -> Dict:
    """
    Standalone function for hybrid search.
    
    Args:
        query_text: Text query for BM25 search
        query_embedding: Vector embedding for semantic search
        top_k: Number of results to return
        client: Optional OpenSearchClient instance (uses default if not provided)
        
    Returns:
        Search results
    """
    if client is None:
        client = _get_default_client()
    response = client.hybrid_search(query_text, query_embedding, top_k); return response.get("hits", {}).get("hits", [])


def semantic_search(
    query_embedding: List[float],
    top_k: int = 5,
    client: Optional[OpenSearchClient] = None,
) -> Dict:
    """
    Standalone function for semantic search.
    
    Args:
        query_embedding: Vector embedding of the query
        top_k: Number of results to return
        client: Optional OpenSearchClient instance
        
    Returns:
        Search results
    """
    if client is None:
        client = _get_default_client()
    return client.semantic_search(query_embedding, top_k)


def keyword_search(
    query_text: str,
    top_k: int = 5,
    client: Optional[OpenSearchClient] = None,
) -> Dict:
    """
    Standalone function for keyword search.
    
    Args:
        query_text: Text query
        top_k: Number of results to return
        client: Optional OpenSearchClient instance
        
    Returns:
        Search results
    """
    if client is None:
        client = _get_default_client()
    return client.keyword_search(query_text, top_k)


def index_document(
    doc_id: str,
    text: str,
    embedding: List[float],
    metadata: Optional[Dict[str, Any]] = None,
    client: Optional[OpenSearchClient] = None,
) -> Dict:
    """
    Standalone function to index a document.
    
    Args:
        doc_id: Document ID
        text: Document text
        embedding: Document embedding
        metadata: Optional metadata
        client: Optional OpenSearchClient instance
        
    Returns:
        Index response
    """
    if client is None:
        client = _get_default_client()
    return client.index_document(doc_id, text, embedding, metadata)


def bulk_index(
    documents: List[Dict[str, Any]],
    client: Optional[OpenSearchClient] = None,
) -> Dict:
    """
    Standalone function for bulk indexing.
    
    Args:
        documents: List of documents
        client: Optional OpenSearchClient instance
        
    Returns:
        Bulk index response
    """
    if client is None:
        client = _get_default_client()
    return client.bulk_index(documents)


def delete_document(
    doc_id: str,
    client: Optional[OpenSearchClient] = None,
) -> Dict:
    """
    Standalone function to delete a document.
    
    Args:
        doc_id: Document ID to delete
        client: Optional OpenSearchClient instance
        
    Returns:
        Delete response
    """
    if client is None:
        client = _get_default_client()
    return client.delete_document(doc_id)


def create_index(
    embedding_dimension: int = 768,
    client: Optional[OpenSearchClient] = None,
) -> bool:
    """
    Standalone function to create the index.
    
    Args:
        embedding_dimension: Dimension of embeddings
        client: Optional OpenSearchClient instance
        
    Returns:
        True if created, False if exists
    """
    if client is None:
        client = _get_default_client()
    return client.create_index(embedding_dimension)
