from FlagEmbedding import FlagModel
import vector_db

def encode(query):
  encoder = FlagModel('BAAI/bge-base-en-v1.5', 
  query_instruction_for_retrieval="Represent this sentence for searching relevant passages:",
  normalize_embeddings=True,
  use_fp16=False)
  
  query_embedding = encoder.encode(query, batch_size=1)
  query_embedding = query_embedding.reshape(1, 768)

  return query_embedding
