import faiss
import json
import gc
import numpy as np
from FlagEmbedding import FlagModel

# encode passed query and return top ids and texts and distances from search
def search(query_embedding, top_k):
  # import index from saved file
  index = faiss.read_index("./index.bin")
  # perform search for 3 most relevant documents for query
  D, I = index.search(query_embedding, k=top_k)
  relevant_indices = I[0][:top_k]
  with open("./data/preprocessed_documents.json", "r") as docs_json: 
    docs = json.load(docs_json)
    texts = [docs[i]["text"] for i in relevant_indices]
  # print top 3 results
  print("\nTop K Results:")
  for rank, idx in enumerate(relevant_indices, start=0):
      print(f"\n{rank + 1}) Document Index: {idx}")
      print(f"Distance: {D[0][rank]}")
      print(f"Text: {texts[rank]}")
  
  return texts