

class RagVer4: 
    """
    improving the retriever 🔎:
    hybrid search (BM25F + semantic search / with reciprocal ranking fusion)
    references:
    - Hybrid Search Explained (Weaviate, 2023): https://weaviate.io/blog/hybrid-search-explained
    - Risk-Reward Trade-offs in Rank Fusion (Benham & Culpepper, 2017): https://rodgerbenham.github.io/bc17-adcs.pdf
    """
    
    def __init__(self):
        # build dtm, upsert vectors, etc.
        pass
    
    def __call__(self, query: str) -> str:
        pass