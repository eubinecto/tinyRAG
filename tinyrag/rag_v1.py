
from pathlib import Path
import spacy
from rank_bm25 import BM25Okapi
import numpy as np
import yaml

class RAGVer1: 
    """
    first attempt at the retriever 🔎:
    keyword search with BM25 scoring
    References: 
    - rank_bm25 - A Collection of BM25 Algorithms in Python (Dorianbrown): https://github.com/dorianbrown/rank_bm25
    - Improved Text Scoring with BM25 (Weber from Elastic, 2016) - https://velog.io/@mayhan/Elasticsearch-유사도-알고리즘
    - TF-IDF와 BM25 사이의 차이점을 잘 정리한 블로그 포스트 - https://velog.io/@mayhan/Elasticsearch-유사도-알고리즘
    """
    
    def __init__(self):
        with open(Path(__file__).resolve().parent / "openai27052023.yaml", 'r') as f:
            self.openai_paper = yaml.safe_load(f)
        # index BM25
        self.nlp = spacy.load("en_core_web_sm")
        self.bm25 = BM25Okapi([[token.lemma_ for token in self.nlp(sent)] for sent in self.openai_paper['sentences']])

        
    def __call__(self, query: str, k: int = 3) -> list[tuple[str, float]]:
        tokens = [token.lemma_ for token in self.nlp(query)]
        sims = self.bm25.get_scores(tokens).tolist()  # (1, N) -> (N,) -> list
        indices = reversed(np.argsort(sims))  # (, E) -> (, E) ->  (E,) -> list
        sims = sorted(sims, reverse=True) # (1, E) -> (1, E) ->  (E,) -> list
        results = [
           (self.openai_paper['sentences'][i], s)
           for i, s in zip(indices, sims)
        ]
        top_k: list[tuple[str, float]] = results[:k]
        return top_k
