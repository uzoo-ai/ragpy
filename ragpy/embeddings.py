from abc import ABC, abstractmethod
from typing import List
from openai import OpenAI
from qdrant_client.models import Document


class EmbeddingService(ABC):
    """
    Abstract base class for embedding generation services.
    """

    @abstractmethod
    def embed_text(self, text: str) -> List[float | Document]:
        """
        Compute embedding for a single text string.
        """
        pass

    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[List[float | Document]]:
        """
        Compute embeddings for a list of text strings.
        """
        pass


class OpenAIEmbeddingService(EmbeddingService):
    """
    Embedding service using OpenAI API.
    """

    def __init__(self, model: str = "text-embedding-3-small", api_key: str = None):
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def embed_text(self, text: str) -> List[float]:
        resp = self.client.embeddings.create(model=self.model, input=text)
        return resp.data[0].embedding

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        resp = self.client.embeddings.create(model=self.model, input=texts)
        return [d.embedding for d in resp.data]
    

class QdrantEmbeddingService(EmbeddingService):
    """
    Embedding service using Qdrant’s built-in embedding models.
    Requires Qdrant >= 1.13 with text-to-vector models enabled.
    """

    def __init__(
            self,
            model: str = "sentence-transformers/all-MiniLM-L6-v2"
        ):
        self.model = model

    def embed_text(self, text: str) -> List[Document]:
        result = Document(text=text, model=self.model)
        return result

    def embed_batch(self, texts: List[str]) -> List[List[Document]]:
        results = [Document(text=t, model=self.model) for t in texts]
        return results
