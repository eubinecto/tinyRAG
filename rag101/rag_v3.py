
from unstructured.documents.elements import Title, NarrativeText
from unstructured.partition.pdf import partition_pdf
from pathlib import Path
import spacy
import weaviate
import os
from .rag_v2 import check_batch_result


class RagVer3: 
    """
    improving the retriever 🔎:
    hybrid search (BM25F + semantic search / with reciprocal ranking fusion)
    references:
    - Hybrid Search Explained (Weaviate, 2023): https://weaviate.io/blog/hybrid-search-explained
    - Risk-Reward Trade-offs in Rank Fusion (Benham & Culpepper, 2017): https://rodgerbenham.github.io/bc17-adcs.pdf
    """
    
    def __init__(self):
                # build dtm, upsert vectors, etc.
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
        ]
        # then ... import sentences into weaviate
        credentials = weaviate.auth.AuthApiKey(os.environ['WEAVIATE_CLUSTER_KEY'])
        self.client = weaviate.Client(
                       os.environ['WEAVIATE_CLUSTER_URL'],
                       credentials,
                       additional_headers={
                            'X-OpenAI-Api-Key': os.environ['OPENAI_API_KEY']
                       }
        )
        # on init, just flush all schema 
        self.client.schema.delete_all()
        # then create a new one
        class_obj = {
            "class": "Sentence",
            "moduleConfig": {
                "text2vec-openai": {
                    "vectorizeClassName": False,
                    "model": "ada",
                    "modelVersion": "002",  #  we are using ada
                    "type": "text"
                }
            },
            "properties": [
                {
                    "name": "content",
                    "dataType": ["text"],
                }
            ],
            "vectorizer": "text2vec-openai"
        }
        self.client.schema.create_class(class_obj)
        with self.client.batch(
            batch_size=3,               # Specify batch size
            num_workers=4,             # Parallelize the process
            dynamic=True,                        # Enable/Disable dynamic batch size change
            timeout_retries=3,           # Number of retries if a timeout occurs
            connection_error_retries=3,  # Number of retries if a connection error occurs
            callback=check_batch_result,
        ) as batch:
            for sent in self.sentences:
                batch.add_data_object(
                    {'content': sent},
                    class_name="Sentence"
                )
    
    def __call__(self, query: str, alpha: float = 0.4, k: int = 3) -> list[tuple[str, float]]:
        # --- search over vectors semantically --- #
        r = (
            self.client.query
            .get("Sentence", ['content'])
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