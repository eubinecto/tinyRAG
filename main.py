from dotenv import load_dotenv
load_dotenv()
import argparse
from tinyrag.rag_v5 import RAGVer5
from pprint import pprint

# init RAG (index sentences)
print("indexing embeddings...")
rag = RAGVer5()
parser = argparse.ArgumentParser()

while True:
    query = input("Question: ")
    answer = rag(query, alpha=0.6)
    pprint(answer, width=70)
