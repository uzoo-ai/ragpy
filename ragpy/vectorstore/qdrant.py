import os
import uuid
import numpy as np
from typing import List, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from ragpy.rag_service import RagService
from ragpy.embeddings import EmbeddingService, QdrantEmbeddingService
from dotenv import load_dotenv

load_dotenv()

QDRANT_API_KEY = os.getenv('QDRANT_API_KEY')


class QdrantService(RagService):
    """
    Qdrant-based implementation of the RagService abstract interface.
    Uses gRPC for high-performance communication.
    """

    def __init__(self, host: str, port: int = 6334, embedder: EmbeddingService = None):
        self.client = QdrantClient(host=host, grpc_port=port, prefer_grpc=True, api_key=QDRANT_API_KEY)
        self.embedder = embedder or QdrantEmbeddingService()

    def create_collection(self, name: str, distance: Distance = Distance.COSINE) -> str:
        embedding_size = None
        match self.embedder.model:
            case 'text-embedding-3-small' | 'text-embedding-ada-002':
                embedding_size = 1536
            case 'text-embedding-3-large':
                embedding_size = 3072
            case _:
                embedding_size = self.client.get_embedding_size(self.embedder.model)
        self.client.recreate_collection(
            collection_name=name,
            vectors_config=VectorParams(size=embedding_size, distance=distance),
        )
        return name
    
    def upsert(self, collection_name: str, chunks: List[str]):
        vectors = self.embedder.embed_batch(chunks)
        payload = [{'document': chunk} for chunk in chunks]
        self.client.upload_collection(collection_name, vectors, payload)

    def search(self, collection_name: str, query_text: str, limit: int = 5) -> List[Dict[str, Any]]:
        query = self.client.query_points(
            collection_name=collection_name,
            query=self.embedder.embed_text(query_text),
            limit=limit,
        )
        return [
            {"id": p.id, "score": p.score, "text": p.payload.get("document")}
            for p in query.points
        ]
