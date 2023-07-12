from .rag_v2 import RAGVer2


class RAGVer3(RAGVer2): 
    """
    improving the retriever 🔎:
    hybrid search (BM25F + semantic search / with reciprocal ranking fusion)
    references:
    - Hybrid Search Explained (Weaviate, 2023): https://weaviate.io/blog/hybrid-search-explained
    - Risk-Reward Trade-offs in Rank Fusion (Benham & Culpepper, 2017): https://rodgerbenham.github.io/bc17-adcs.pdf
    """
    
    def __call__(self, query: str, alpha: float = 0.4, k: int = 3) -> list[tuple[str, float]]:
        # --- search over vectors semantically --- #
        r = (
            self.client.query
            .get("Sentence", ['content'])
            # alpha; 0.0 = only BM25F, 1.0 = only semantic search
            .with_hybrid(query=query, alpha=alpha)
            .with_limit(k)
            .with_additional(["score", "explainScore"])
            .do()
        )
        results =  list()
        for res in r['data']['Get']['Sentence']:
            sent = res['content']
            sim = res['_additional']['score']
            results.append((sent, sim))
        return results