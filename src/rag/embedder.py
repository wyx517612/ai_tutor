
import os
from sentence_transformers import SentenceTransformer

# 设置缓存到临时目录（部署环境可用）
os.environ["HF_HOME"] = "/tmp/huggingface"
os.environ["TRANSFORMERS_CACHE"] = "/tmp/huggingface"

# 用轻量级多语言模型（支持中文，约 120MB）
embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

def embed_chunk(text):
    """把文本转换成向量"""
    return embedding_model.encode(text)