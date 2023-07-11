
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from unstructured.documents.elements import Title, NarrativeText
from unstructured.partition.pdf import partition_pdf
from pathlib import Path
import spacy


class RagVer1: 
    """
    first attempt at the retriever 🔎:
    keyword search with TFIDF scoring + unstructured io + stuffing answer generator 
    references:
    - Scoring, term weighting and the vector space model (Manning et al., 2008, Stanford, chatper 6 of the book "Information Retreival"): https://nlp.stanford.edu/IR-book/html/htmledition/scoring-term-weighting-and-the-vector-space-model-1.html
    - scikit-learn; TfidfVectorizer (Scikit-learn , 2023): https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
    - PDF is an unstructured format - https://stackoverflow.com/a/57443276
    - Why Text Extraction is Hard (PyPDF2, 2006): https://pypdf2.readthedocs.io/en/latest/user/extract-text.html
    - Unstructured IO -  Open source libraries and APIs to build custom preprocessing pipelines for labeling, training, or production machine learning pipelines. : https://github.com/Unstructured-IO/unstructured
    - Combining LangChain and Weaviate (Weaviate, 2023): https://weaviate.io/blog/combining-langchain-and-weaviate
    """
    
    def __init__(self):
        elements = partition_pdf(filename=Path(__file__).resolve().parent.parent / "openai27052023.pdf", strategy="auto")
        paragraphs = ""
        for el in elements:
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
        ] # get sentences ... connected as an ngram 
        # then build a TFIDF model
        self.tfidf_vectorizer: TfidfVectorizer = TfidfVectorizer(tokenizer=lambda x: [token.lemma_ for token in self.nlp(x)])
        # a sparse matrix, really
        self.embeddings = self.tfidf_vectorizer.fit_transform(self.sentences)
    
    
    def __call__(self, query: str, k: int = 3) -> list[str, float]:
        #  get the query embedding
        embedding = self.tfidf_vectorizer.transform([query])  # (1, |V|)
        sims = cosine_similarity(embedding, self.embeddings).squeeze().tolist()  # (1, N)
        indices = reversed((np.argsort(sims)))
        sims = sorted(sims, reverse=True)
        results = [
           (self.sentences[i], s)
           for i, s in zip(indices, sims)
        ]
        # just get top_k
        top_k: list[tuple[str, float]] = results[:k]
        # put them in a prompt (stuffing)
        return top_k