from .rag_v3 import RAGVer3
from unstructured.documents.elements import Title
import openai


class RAGVer4(RAGVer3):
    """
    first attempt at the reader 📖:    
    References:
    - Stuffiing (Weaviate, 2023): https://weaviate.io/blog/combining-langchain-and-weaviate
    """

    def __init__(self):
        super().__init__()
        self.extract_title() # plus - extract titles

    def extract_title(self):
        title_element = [el for el in self.elements if isinstance(el, Title)][0]
        self.title = str(title_element).strip()
    

    def __call__(self, query: str, alpha: float = 0.4, k: int = 3) -> tuple[str, list[tuple[str, float]]]:
        results: list[tuple[str, float]] = super().__call__(query, alpha, k)
        # with this, generate an answer 
        excerpts = [res[0] for res in results]
        excerpts = '\n'.join([f'[{i}]. {excerpt}' for i, excerpt in enumerate(excerpts, start=1)])
        prompt = f"""
        user query:
        {query}
        
        title of the paper:
        {self.title}
        
        excerpts: 
        {excerpts}
        ---
        given the excerpts from the paper above, answer the user query.
        In your answer, make sure to cite the excerpts by its number wherever appropriate.
        Note, however, that the excerpts may not be relevant to the user query.
        """
        chat_completion = openai.ChatCompletion.create(model="gpt-3.5-turbo",
                                                       messages=[{"role": "user", "content": prompt}])
        answer = chat_completion.choices[0].message.content
        return answer, results
    