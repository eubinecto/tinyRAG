"""
https://github.com/dorianbrown/rank_bm25
"""

from rank_bm25 import BM25Okapi

corpus = [
    "Hello there good man!",
    "It is quite windy in London",
    "How is the weather today ? sunny ? windy ?"
]

tokenized_corpus = [doc.split(" ") for doc in corpus]
bm25 = BM25Okapi(tokenized_corpus)

query = "windy"
tokenized_query = query.split(" ")

from pprint import pprint

pprint(bm25.get_scores(tokenized_query))

"""
array([0.        , 0.10029299, 0.0781876 ])
"""

pprint(bm25.get_top_n(tokenized_query, corpus, n=len(corpus)))

"""
['It is quite windy in London',
 'How is the weather today ? sunny ? windy ?',
 'Hello there good man!']
"""