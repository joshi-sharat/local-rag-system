class EmbeddingService:
    def __init__(self, model_path=EMBEDDING_MODEL_PATH)
    def encode(self, text: str) -> List[float]
    def encode_batch(self, texts: List[str]) -> List[List[float]]
    def get_dimension(self) -> int