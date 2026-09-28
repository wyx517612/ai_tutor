import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from sentence_transformers import SentenceTransformer

embedding_model = SentenceTransformer("shibing624/text2vec-base-chinese")

def embed_chunk(text):
    return embedding_model.encode(text).tolist()