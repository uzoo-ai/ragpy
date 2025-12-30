from abc import ABC, abstractmethod
from typing import List, Any, Dict


class RagService(ABC):
    """
    Abstract base class for RAG (Retrieval-Augmented Generation) vector stores.
    Concrete implementations: Qdrant, Weaviate, Chroma, Pinecone, etc.
    """

    @abstractmethod
    def create_collection(self, name: str) -> str:
        """
        Create a collection (namespace, index, etc.) in the vector store.

        :param name: Name of the collection
        :return: Collection ID or name
        """
        pass

    @abstractmethod
    def upsert(self, collection_name: str, texts: List[str]) -> None:
        """
        Insert or update text chunks with their embeddings into a collection.

        :param collection_name: Target collection name
        :param texts: List of text chunks
        """
        pass

    @abstractmethod
    def search(self, collection_name: str, query_vector: List[float], limit: int = 5) -> List[Dict[str, Any]]:
        """
        Perform a vector similarity search.

        :param collection_name: Name of the collection
        :param query_vector: Query embedding vector
        :param limit: Maximum number of results to return
        :return: List of search results
        """
        pass
