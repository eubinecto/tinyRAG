from rag101.rag_v4 import RAGVer4
from pprint import pprint
from dotenv import load_dotenv
load_dotenv()
rag = RAGVer4()


answer, excerpts = rag("What was the main objective of the paper?", alpha=0.6)
print(answer)
print("----")
pprint(excerpts)


print("######")
answer, excerpts = rag("When was the paper published?", alpha=0.6)  #  asking for some facts
print(answer)
print("----")
pprint(excerpts)



print("######")
answer, excerpts = rag("How did the authors tested GPT4?", alpha=0.6)
print(answer)
print("----")
pprint(excerpts)




print("######")
answer, excerpts = rag("In what ways GPT4 are limited by?", alpha=0.6)
print(answer)
print("----")
pprint(excerpts)



print("######")
answer, excerpts = rag("What is the architecture of GPT4?", alpha=0.6)
print(answer)
print("----")
pprint(excerpts)



print("######")
answer, excerpts = rag("Does GPT4 demonstrate near-human intelligence?", alpha=0.6)
print(answer)
print("----")
pprint(excerpts)

