from rag101.rag_v2 import RagVer2
from pprint import pprint


rag = RagVer2()


# good for searching for keywords
pprint(rag("the main goal"))


# but not good for answering questions
print("######")
pprint(rag("what is the main goal of the paper?"))
