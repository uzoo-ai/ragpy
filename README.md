## RAGpy

### Usage

Create a vectorstore client (currently only Qdrant is supported).
Set QDRANT_API_KEY environment variable first.

```
from ragpy.vectorstore import QdrantService
from ragpy.embeddings import OpenAIEmbeddingService

embedder = OpenAIEmbeddingService(api_key="your-openai-key")
qdrant = QdrantService("https://your-qdrant-server-url", embedder=embedder)
```

Create collection:

```
qdrant.create_collection('collection-name')
```

Upload chunks of data to a collection:

```
chunks = ['list of', 'text chunks']
qdrant.upsert('collection-name', chunks)
```

Query collection:

```
query_points = qdrant.search('collection-name', 'your query')
for point in query_points:
    print(f'score: {point["score"]}')
    print(f'text: {point["text"]}')
```