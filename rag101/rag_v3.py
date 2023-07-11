

class RagVer3: 
    """
    improving the retriever 🔎:
    semantic search with Approximate Neartest Neighbour (ANN) search
    References:
    - Vector Indexing (Weaviate, 2023): https://weaviate.io/developers/weaviate/concepts/vector-index
    - Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs (Malkov & Yashunin, 2018): https://arxiv.org/abs/1603.09320
    - text2vec-openai (Weaviate, 2023): https://weaviate.io/developers/weaviate/modules/retriever-vectorizer-modules/text2vec-openai
    - (Batch) Import items (Weaviate, 2023): https://weaviate.io/developers/weaviate/manage-data/import
    """
    
    def __init__(self):
        # build dtm, upsert vectors, etc.
        pass
    
    def __call__(self, query: str) -> str:
        pass