import os

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from sentence_transformers import SentenceTransformer, cross_encoder


embedding_model=SentenceTransformer("shibing624/text2vec-base-chinese")
def embed_chunk(chunk:str) -> list[float]:
    embedding=embedding_model.encode(chunk)
    return embedding.tolist()