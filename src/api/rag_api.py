# Endpoints needed:
POST /api/search          # Search documents with hybrid search
POST /api/documents       # Upload and index documents  
GET  /api/documents       # List indexed documents
DELETE /api/documents/{id} # Delete a document
POST /api/embeddings      # Generate embeddings for text
GET  /api/health          # Health check
POST /api/yoga-class      # YogaBharati-specific: Get yoga class context