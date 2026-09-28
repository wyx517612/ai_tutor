
import os

# 1. 先设置缓存目录（必须在任何相关 import 之前）
os.environ["HF_HOME"] = "C:/Users/w1850/.cache/huggingface"  # 换回你原来的缓存路径
os.environ["TRANSFORMERS_CACHE"] = "C:/Users/w1850/.cache/huggingface"

# 2. 再 import
from sentence_transformers import SentenceTransformer

embedding_model = SentenceTransformer("models/text2vec-base-chinese")
def embed_chunk(text):
    """把文本转换成向量"""
    return embedding_model.encode(text)