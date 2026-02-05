"""
Set up OpenSearch index and hybrid search pipeline.
Run once before starting the service.
"""
import requests
from src.constants import (
    OPENSEARCH_HOST,
    OPENSEARCH_PORT,
    OPENSEARCH_INDEX,
    EMBEDDING_DIMENSION,
)


def setup_hybrid_search_pipeline():
    """Create the NLP search pipeline for hybrid search."""
    url = f"http://{OPENSEARCH_HOST}:{OPENSEARCH_PORT}/_search/pipeline/nlp-search-pipeline"

    pipeline = {
        "description": "Post processor for hybrid search",
        "phase_results_processors": [
            {
                "normalization-processor": {
                    "normalization": {"technique": "min_max"},
                    "combination": {
                        "technique": "arithmetic_mean",
                        "parameters": {"weights": [0.3, 0.7]},
                    },
                }
            }
        ],
    }

    response = requests.put(url, json=pipeline)
    if response.status_code == 200:
        print("✅ Hybrid search pipeline created successfully")
    else:
        print(f"❌ Failed to create pipeline: {response.status_code} - {response.text}")
    return response.status_code == 200


def create_index():
    """Create the document index with KNN mapping."""
    base_url = f"http://{OPENSEARCH_HOST}:{OPENSEARCH_PORT}"

    # Check if index already exists
    check_url = f"{base_url}/{OPENSEARCH_INDEX}"
    check_response = requests.head(check_url)

    if check_response.status_code == 200:
        print(f"⚠️  Index '{OPENSEARCH_INDEX}' already exists. Skipping creation.")
        return True

    # Index configuration with KNN settings
    index_config = {
        "settings": {
            "index": {
                "number_of_shards": 1,
                "number_of_replicas": 0,
                "knn": True,
            }
        },
        "mappings": {
            "properties": {
                "text": {
                    "type": "text",
                    "analyzer": "standard",
                },
                "embedding": {
                    "type": "knn_vector",
                    "dimension": EMBEDDING_DIMENSION,
                    "method": {
                        "engine": "faiss",
                        "space_type": "l2",
                        "name": "hnsw",
                        "parameters": {
                            "ef_construction": 256,
                            "m": 48,
                        },
                    },
                },
                "document_name": {
                    "type": "keyword",
                },
                "chunk_id": {
                    "type": "keyword",
                },
                "metadata": {
                    "type": "object",
                    "enabled": True,
                },
            }
        },
    }

    # Create the index
    create_url = f"{base_url}/{OPENSEARCH_INDEX}"
    response = requests.put(
        create_url,
        json=index_config,
        headers={"Content-Type": "application/json"},
    )

    if response.status_code == 200:
        print(f"✅ Index '{OPENSEARCH_INDEX}' created successfully")
        print(f"   - Embedding dimension: {EMBEDDING_DIMENSION}")
        print(f"   - KNN engine: FAISS with HNSW algorithm")
        return True
    else:
        print(f"❌ Failed to create index: {response.status_code} - {response.text}")
        return False


def delete_index():
    """Delete the document index (useful for resetting)."""
    url = f"http://{OPENSEARCH_HOST}:{OPENSEARCH_PORT}/{OPENSEARCH_INDEX}"

    response = requests.delete(url)
    if response.status_code == 200:
        print(f"✅ Index '{OPENSEARCH_INDEX}' deleted successfully")
        return True
    elif response.status_code == 404:
        print(f"⚠️  Index '{OPENSEARCH_INDEX}' does not exist")
        return True
    else:
        print(f"❌ Failed to delete index: {response.status_code} - {response.text}")
        return False


def check_opensearch_connection():
    """Verify OpenSearch is running and accessible."""
    url = f"http://{OPENSEARCH_HOST}:{OPENSEARCH_PORT}"

    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            info = response.json()
            print(f"✅ Connected to OpenSearch")
            print(f"   - Cluster: {info.get('cluster_name', 'unknown')}")
            print(f"   - Version: {info.get('version', {}).get('number', 'unknown')}")
            return True
    except requests.exceptions.ConnectionError:
        print(f"❌ Cannot connect to OpenSearch at {OPENSEARCH_HOST}:{OPENSEARCH_PORT}")
        print("   Make sure OpenSearch is running (e.g., via docker-compose)")
        return False
    except requests.exceptions.Timeout:
        print(f"❌ Connection to OpenSearch timed out")
        return False

    return False


if __name__ == "__main__":
    print("=" * 60)
    print("OpenSearch Setup Script")
    print("=" * 60)
    print()

    # Step 1: Check connection
    print("Step 1: Checking OpenSearch connection...")
    if not check_opensearch_connection():
        print("\nSetup aborted. Please start OpenSearch first.")
        exit(1)
    print()

    # Step 2: Create index
    print("Step 2: Creating document index...")
    if not create_index():
        print("\nSetup aborted due to index creation failure.")
        exit(1)
    print()

    # Step 3: Setup hybrid search pipeline
    print("Step 3: Setting up hybrid search pipeline...")
    if not setup_hybrid_search_pipeline():
        print("\nWarning: Pipeline creation failed, but index was created.")
        print("Hybrid search may not work correctly.")
    print()

    print("=" * 60)
    print("Setup complete!")
    print("=" * 60)