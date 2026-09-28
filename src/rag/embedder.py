import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from sentence_transformers import SentenceTransformer

embedding_model = SentenceTransformer("BAAI/bge-small-zh-v1.5")

def embed_chunk(text):
    return embedding_model.encode(text).tolist()