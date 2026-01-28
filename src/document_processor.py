class DocumentProcessor:
    def __init__(self, embedding_service, opensearch_client)
    def process_pdf(self, file_path: str) -> str  # Uses OCR
    def chunk_text(self, text: str, chunk_size: int) -> List[str]
    def process_and_index(self, content, document_name, metadata) -> int