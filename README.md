# tinyRAG

A minimal implementation of a retriever-augmented generation (RAG) system. 
 
## What is this? 


`tinyrag` answers any questions on *GPT Technical Report* (OpenAI, 2023) with in-text citations | 
--- | 
<img width="1120" alt="image" src="https://github.com/eubinecto/tinyRAG/assets/56193069/e26c4030-23ae-44a4-93f5-55641a04da7a"> | 

like so:
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

It is my attempt to reverse-engineer RAG systems - e.g. Perplexity AI, Bing Chat, etc - in the simplest way possible. 
Mostly to educate myself with the theoretical and technical aspects of RAG. 

## Quick Start 🚀

Install `poetry`: 

```bash 
curl -sSL https://install.python-poetry.org | python3 -
```

Clone the project: 
```
git clone https://github.com/eubinecto/tinyRAG.git
```

Install dependencies:
```
cd tinyrag
poetry install
```

Now go create your own `weaviate` cluster.  Visit [Weaviate Cloud Services](https://weaviate.io/pricing), login to the console, Create a cluster ("Free Sandbox" should be free for 14days). 

Press "Details", and take a note of two credentials: "Cluster URL" & and your Cluster API Key.  

Type them in a `.env` file, along with your [OpenAI Key](https://www.howtogeek.com/885918/how-to-get-an-openai-api-key/), and put them under the root directory. Keep your`.env` in the following format: 

```env
WEAVIATE_CLUSTER_KEY=<your cluster api key>
WEAVIATE_CLUSTER_URL=<your cluster url>
OPENAI_API_KEY=<your openai api key>
```

That's it for logistics. Now you can try asking questions like so:

```python
from tinyrag.rag_v5 import RAGVer5
from dotenv import load_dotenv
load_dotenv() 
rag = RAGVer5()
print("######")
answer = rag("Does GPT4 demonstrate near-human intelligence?", alpha=0.6)
print(answer)
```
```
Based on the given excerpts from the paper "GPT-4 Technical Report,"
there is evidence that GPT-4 demonstrates near-human intelligence. 

Excerpt [1] states that GPT-4 was evaluated on exams designed for humans and performs quite well,
often outscoring the majority of human test takers.
This suggests that GPT-4 exhibits a level of intelligence that is comparable to or even surpasses humans in certain scenarios.

Excerpt [2] further supports this, stating that GPT-4 exhibits human-level performance
on various professional and academic benchmarks,
including passing a simulated bar exam with a score among the top 10% of test takers. 
This indicates that in these specific domains, GPT-4 can perform at a level comparable to that of human experts.

It should be noted, however, that both excerpts [2] and [3] also mention that
 GPT-4 is "less capable than humans in many real-world scenarios."
This suggests that while GPT-4 may demonstrate near-human intelligence in specific domains, it may not possess the same level of general intelligence or adaptability as humans.
--- EXCERPTS ---
[1]. "To test its capabilities in such scenarios, GPT-4 was evaluated on a variety of exams originally designed for humans. In these evaluations it performs quite well and often outscores the vast majority of human test takers."
[2]. "While less capable than humans in many real-world scenarios, GPT-4 exhibits human-level performance on various professional and academic benchmarks, including passing a simulated bar exam with a score around the top 10% of test takers. GPT-4 is a Transformer- based model pre-trained to predict the next token in a document."
[3]. "We report the development of GPT-4, a large-scale, multimodal model which can accept image and text inputs and produce text outputs. While less capable than humans in many real-world scenarios, GPT-4 exhibits human-level performance on various professional and academic benchmarks, including passing a simulated bar exam with a score around the top 10% of test takers."
```

## How it's made - the Retriever 🔎

### `RAGVer1` - searching for keywords with BM25 scoring
relevant references: 
- rank_bm25 - A Collection of BM25 Algorithms in Python (Dorianbrown): https://github.com/dorianbrown/rank_bm25
- Improved Text Scoring with BM25 (Weber from Elastic, 2016) - https://velog.io/@mayhan/Elasticsearch-유사도-알고리즘


example output (good):
https://github.com/eubinecto/tinyRAG/blob/37a695ee8dc652e79a72eab89da3146ce285d6c1/main_rag_v1.py#L8-L31

example output (bad):
https://github.com/eubinecto/tinyRAG/blob/37a695ee8dc652e79a72eab89da3146ce285d6c1/main_rag_v1.py#L33-L50



### `RAGVer2` - searching for meaning with ANN (Approximate Nearest Neighbor) 

relevant references:
- Vector Indexing (Weaviate, 2023): https://weaviate.io/developers/weaviate/concepts/vector-index
- Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs (Malkov & Yashunin, 2018): https://arxiv.org/abs/1603.09320
- text2vec-openai (Weaviate, 2023): https://weaviate.io/developers/weaviate/modules/retriever-vectorizer-modules/text2vec-openai
- (Batch) Import items (Weaviate, 2023): https://weaviate.io/developers/weaviate/manage-data/import


example output (good): 
https://github.com/eubinecto/tinyRAG/blob/37a695ee8dc652e79a72eab89da3146ce285d6c1/main_rag_v2.py#L41-L58


example output (bad): 
https://github.com/eubinecto/tinyRAG/blob/37a695ee8dc652e79a72eab89da3146ce285d6c1/main_rag_v2.py#L9-L23


### `RAGVer3` - brining the best of both worlds - hybrid search with RRF (Reciprocal Rank Fusion)

relevant references:
- reciprocal rank fusion (elastic, 2023) - https://www.elastic.co/guide/en/elasticsearch/reference/current/rrf.html
- weaviate - hybrid search explained (weaviate, 2023) - https://weaviate.io/blog/hybrid-search-explained 

example (good at keyword search):
https://github.com/eubinecto/tinyRAG/blob/ece84fbf01e7f0368d61b1203623bd79e5fb3397/main_rag_v3.py#L9-L27

example (not bad at semantic search): 
https://github.com/eubinecto/tinyRAG/blob/ece84fbf01e7f0368d61b1203623bd79e5fb3397/main_rag_v3.py#L47-L62


## How it's made - The Reader 📖

### `RAGVer4` - generating answers with stuffing

relevant literature:
- weaviate - stuffing - https://weaviate.io/blog/combining-langchain-and-weaviate

example output (good): 
https://github.com/eubinecto/tinyRAG/blob/ece84fbf01e7f0368d61b1203623bd79e5fb3397/main_rag_v4.py#L44-L57


example output (bad):
https://github.com/eubinecto/tinyRAG/blob/ece84fbf01e7f0368d61b1203623bd79e5fb3397/main_rag_v4.py#L18-L40

### `RAGVer5` - moderating answers with Chain-of-Thought & guidance

relevant literature: 
- Chain-of-Thought Prompting Elicits Reasoning in Large Language Models (Wei et al., 2023) - https://arxiv.org/abs/2201.11903
- A guidance language for controlling large language models (Microsoft, 2023) - https://github.com/microsoft/guidance

example output (good at being tentative when needed): 
https://github.com/eubinecto/tinyRAG/blob/ece84fbf01e7f0368d61b1203623bd79e5fb3397/main_rag_v5.py#L6-L22



  

