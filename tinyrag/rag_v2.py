from tqdm import tqdm
import weaviate
import os
from unstructured.partition.pdf import partition_pdf
from pathlib import Path
from .rag_v1 import RAGVer1

# --- upload corpus as a batch --- #
def check_batch_result(results: dict):
    """
    Check batch results for errors.

    Parameters
    ----------
    results : dict
        The Weaviate batch creation return value.
    """
    if results is not None:
        for result in tqdm(results):
            if "result" in result and "errors" in result["result"]:
                if "error" in result["result"]["errors"]:
                    print(result["result"])


class RAGVer2(RAGVer1): 
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
        self.elements = partition_pdf(filename=Path(__file__).resolve().parent / "openai27052023.pdf", strategy="auto")
        self.extract_sentences()
        self.extract_title()
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
        
        
    
    def __call__(self, query: str, k: int = 3) -> list[tuple[str, float]]:
        # --- search over vectors semantically --- #
        r = (
            self.client.query
            .get("Sentence", ['content'])
            .with_near_text({'concepts': query})
            .with_limit(k)
            .with_additional(["distance"])
            .do()
        )
        results =  list()
        for res in r['data']['Get']['Sentence']:
            sent = res['content']
            cosine_distance = res['_additional']['distance']
            cosine_sim = 1 - cosine_distance
            results.append((sent, cosine_sim))
        return results