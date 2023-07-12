
from unstructured.documents.elements import Title, NarrativeText
from unstructured.partition.pdf import partition_pdf
from pathlib import Path
import spacy
from rank_bm25 import BM25Okapi
import numpy as np


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
        self.elements = partition_pdf(filename=Path(__file__).resolve().parent/ "openai27052023.pdf", strategy="auto")
        self.extract_sentences()
        self.extract_title()
        self.bm25 = BM25Okapi([[token.lemma_ for token in self.nlp(sent)] for sent in self.sentences])

    def extract_sentences(self):
        # build dtm, upsert vectors, etc.
        paragraphs = ""
        for el in self.elements:
            if isinstance(el, Title):
                paragraphs += "<TITLE>"
            if isinstance(el, NarrativeText):
                el_as_str = str(el).strip()
                if " " in el_as_str and not el_as_str.startswith("["):
                    paragraphs += el_as_str
        paragraphs = [p for p in paragraphs.split("<TITLE>") if p]
        self.nlp = spacy.load("en_core_web_sm")
        sentences_by_paragraph: list[list[str]] = [
            [sent.text for sent in self.nlp(p).sents]
            for p in paragraphs
        ]
        bigrams_by_paragraph: list[list[str]] = [
            [f"{sentences[i]} {sentences[i+1]}" for i in range(len(sentences)-1)]
            for sentences in sentences_by_paragraph
        ]
        # just flatten it out
        self.sentences: list[str] = [
            sent
            for sentences in bigrams_by_paragraph
            for sent in sentences
        ]

    def extract_title(self):
        title_element = [el for el in self.elements if isinstance(el, Title)][0]
        self.title = str(title_element).strip()
        
        
    def __call__(self, query: str, k: int = 3) -> list[tuple[str, float]]:
        tokens = [token.lemma_ for token in self.nlp(query)]
        sims = self.bm25.get_scores(tokens).tolist()  # (1, N) -> (N,) -> list
        indices = reversed(np.argsort(sims))  # (, E) -> (, E) ->  (E,) -> list
        sims = sorted(sims, reverse=True) # (1, E) -> (1, E) ->  (E,) -> list
        results = [
           (self.sentences[i], s)
           for i, s in zip(indices, sims)
        ]
        top_k: list[tuple[str, float]] = results[:k]
        return top_k
