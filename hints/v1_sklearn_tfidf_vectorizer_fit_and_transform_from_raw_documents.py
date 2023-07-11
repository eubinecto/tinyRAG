from sklearn.feature_extraction.text import TfidfVectorizer
import spacy

corpus = [
     'This is the first document.',
     'This document is the second document.',
     'And this is the third one.',
     'Is this the first document?',
]


nlp = spacy.load("en_core_web_sm")
vectorizer = TfidfVectorizer(tokenizer=lambda x: [token.lemma_ for token in nlp(x)])
embeddings = vectorizer.fit_transform(corpus)  #  (N, |V|)

# note - these are sparse vectors
print(embeddings.todense())
"""
[[0.42520648 0.         0.         0.34763416 0.42520648 0.5252146
  0.         0.         0.34763416 0.         0.34763416]
 [0.32513203 0.         0.         0.26581674 0.65026407 0.
  0.         0.50938216 0.26581674 0.         0.26581674]
 [0.31055267 0.         0.48654076 0.25389715 0.         0.
  0.48654076 0.         0.25389715 0.48654076 0.25389715]
 [0.         0.59276931 0.         0.30933162 0.37835697 0.46734613
  0.         0.         0.30933162 0.         0.30933162]]
"""
print(embeddings.shape)  #  (N, |V|)
"""
(4, 11)
"""

vocbulary = vectorizer.get_feature_names_out()
print(vocbulary)
print(len(vocbulary))
assert embeddings.shape[1] == len(vocbulary)
"""
['.' '?' 'and' 'be' 'document' 'first' 'one' 'second' 'the' 'third' 'this']
11
"""


query = "is this a document?"
query_embedding = vectorizer.transform([query])  #  (1, |V|)


from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

sims = cosine_similarity(query_embedding, embeddings)  # (1, N)
print(sims)

"""
[[0.45393874 0.49563763 0.1896624  0.82819173]]
"""

indices = np.fliplr(np.argsort(sims, axis=1)).squeeze().tolist()  # (1, E) -> (1, E) ->  (E,) -> list
sims = np.fliplr(np.sort(sims, axis=1)).squeeze().tolist() # (1, E) -> (1, E) ->  (E,) -> list
results = [
    (corpus[i], s)
    for i, s in zip(indices, sims)
]

from pprint import pprint
print(f"query: {query}")
pprint(results)

"""
query: is this a document?
[('Is this the first document?', 0.8281917325067822),
 ('This document is the second document.', 0.49563762902566294),
 ('This is the first document.', 0.453938735722099),
 ('And this is the third one.', 0.18966240275139387)]
"""
