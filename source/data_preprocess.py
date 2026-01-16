from FlagEmbedding import FlagModel
import json
import torch

device = "mps" if torch.backends.mps.is_available() else "cpu"

model = FlagModel(
    'BAAI/bge-base-en-v1.5',
    query_instruction_for_retrieval="Represent this sentence for searching relevant passages:",
    normalize_embeddings=True,
    use_fp16=True
)

# check what device model is running on
model.model = model.model.to(device)
print(f"Running on: {device}")

# read documents.json into docs object 
with open("./data/documents.json", "r") as docs_json: 
    docs = json.load(docs_json)

# extract all texts for encoding and collect embeddings (batch size 64)
texts = [doc["text"] for doc in docs]

embeddings = model.encode(texts, batch_size=64, convert_to_numpy=False)

# build preprocessed list of dicts now with encodings 
preprocessed = [ 
    {"id": doc["id"], "text": doc["text"], "embedding": emb.tolist()} 
    for doc, emb in zip(docs, embeddings)
]

# write to preprocessed_documents.json
output_path = "./data/preprocessed_documents.json"
with open(output_path, "w") as f:
    json.dump(preprocessed, f, indent=4, ensure_ascii=False)

print("Complete. Find preprocessed documents at ./data/preprocessed_documents.json")