import os
from typing import List
from sentence_transformers import SentenceTransformer

# 从项目内的 models 目录加载，不走网络
embedding_model = SentenceTransformer("./models/text2vec-base-chinese")

def embed_chunk(chunk: str) -> List[float]:
    embedding = embedding_model.encode(chunk)
    return embedding.tolist()