from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import torch

# Choose a small, efficient model (change as needed)
MODEL_NAME = "BAAI/bge-small-en-v1.5"  # or "thenlper/gte-small", "sentence-transformers/all-MiniLM-L6-v2"

# Load model (uses GPU if available)
model = SentenceTransformer(MODEL_NAME, device="cuda" if torch.cuda.is_available() else "cpu")

# Example corpus
documents = [
    "The Amazon rainforest is the largest tropical rainforest in the world.",
    "The Mughal Empire was a powerful dynasty in India.",
    "Conservation efforts are essential for the Amazon ecosystem.",
    "Akbar the Great was a notable Mughal emperor.",
    "The Taj Mahal is a famous example of Mughal architecture."
]

# Encode documents to 384-d vectors
doc_embeddings = model.encode(documents, convert_to_numpy=True, normalize_embeddings=True)

# Build FAISS index for fast similarity search (L2 or cosine)
d = doc_embeddings.shape[1]
index = faiss.IndexFlatIP(d)  # Inner product (cosine if vectors are normalized)
index.add(doc_embeddings)

# Query example
query = "Who built the Taj Mahal?"
query_vec = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)

# Search top 3 most similar documents
k = 3
scores, indices = index.search(query_vec, k)
print("Query:", query)
for i, idx in enumerate(indices[0]):
    print(f"Rank {i+1}: {documents[idx]} (score: {scores[0][i]:.4f})")

# --- For PostgreSQL/PGVector IVFFlat (not runnable here, but for reference) ---
# 1. Create table:
# CREATE TABLE docs (id serial PRIMARY KEY, content text, embedding vector(384));
# 2. Insert vectors (use psycopg2 or SQLAlchemy to insert numpy arrays as lists)
# 3. Create IVFFlat index:
# CREATE INDEX ON docs USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
# 4. Query:
# SELECT id, content, embedding <=> '[your_query_vector]' AS distance FROM docs ORDER BY distance LIMIT 3;

# Notes:
# - For large datasets, use PGVector with IVFFlat for disk-based, resource-efficient search.
# - For even more efficiency, consider ONNX export and quantization.
# - Use batch encoding for throughput, but keep batch size small on 4GB VRAM.