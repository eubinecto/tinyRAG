# tinyRAG

A minimal implementation of a retriever-augmented generation (RAG) system. 
 
## What is this? 

```python
from tinyrag.rag_v5 import RAGVer5
from dotenv import load_dotenv
load_dotenv()
rag = RAGVer5()
answer = rag("In what ways is GPT4 limited by?", alpha=0.6)
print(answer)
```
```text
"""
GPT-4 is limited in several ways, as mentioned in the paper "GPT-4 Technical Report" [1][2][3]. Despite its capabilities, GPT-4 still suffers from limitations similar to earlier GPT models. One major limitation is its lack of full reliability, as it may "hallucinate" facts and make reasoning errors [1]. Additionally, GPT-4 has a limited context window, which restricts its understanding and processing of larger bodies of text [2]. These limitations pose significant and novel safety challenges, highlighting the need for extensive research in areas like bias, disinformation, over-reliance, privacy, cybersecurity, and proliferation [3].
--- EXCERPTS ---
[1]. "Despite its capabilities, GPT-4 has similar limitations as earlier GPT models. Most importantly, it still is not fully reliable (it “hallucinates” facts and makes reasoning errors)."
[2]. "Despite its capabilities, GPT-4 has similar limitations to earlier GPT models [1, 37, 38]: it is not fully reliable (e.g. can suffer from “hallucinations”), has a limited context window, and does not learn∗Please cite this work as “OpenAI (2023)". Full authorship contribution statements appear at the end of thedocument."
[3]. "GPT-4’s capabilities and limitations create signiﬁcant and novel safety challenges, and we believe careful study of these challenges is an important area of research given the potential societal impact. This report includes an extensive system card (after the Appendix) describing some of the risks we foresee around bias, disinformation, over-reliance, privacy, cybersecurity, proliferation, and more."
"""
```

It is my attempt to reverse-engineer popular RAG systems, e.g. Perplexity AI, Bing Chat, etc, in the most simple way possible. 


## The Retriever 🔎

### `RAGVer1` - searching for keywords with BM25 scoring

```


```



### `RAGVer2` - searching for meaning with ANN (Approximate Nearest Neighbor)

```


```


### `RAGVer3` - brining the best of both worlds - hybrid search with RRF (Reciprocal Rank Fusion)

```


```


## The Reader 📖

### `RAGVer4` - generating answers with stuffing

```


```

### `RAGVer5` - moderating answers with Chain-of-Thought & guidance

```

```




