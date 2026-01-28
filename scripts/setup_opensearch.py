"""
Set up OpenSearch index and hybrid search pipeline.
Run once before starting the service.
"""
import requests
from src.constants import OPENSEARCH_HOST, OPENSEARCH_PORT, OPENSEARCH_INDEX

def setup_hybrid_search_pipeline():
    """Create the NLP search pipeline for hybrid search."""
    url = f"http://{OPENSEARCH_HOST}:{OPENSEARCH_PORT}/_search/pipeline/nlp-search-pipeline"
    
    pipeline = {
        "description": "Post processor for hybrid search",
        "phase_results_processors": [{
            "normalization-processor": {
                "normalization": {"technique": "min_max"},
                "combination": {
                    "technique": "arithmetic_mean",
                    "parameters": {"weights": [0.3, 0.7]}
                }
            }
        }]
    }
    
    response = requests.put(url, json=pipeline)
    return response.status_code == 200

def create_index():
    """Create the document index with KNN mapping."""
    # Implementation...

if __name__ == "__main__":
    setup_hybrid_search_pipeline()
    create_index()