
import os

# 1. 先设置缓存目录（必须在任何相关 import 之前）
os.environ["HF_HOME"] = "C:/Users/w1850/.cache/huggingface"  # 换回你原来的缓存路径
os.environ["TRANSFORMERS_CACHE"] = "C:/Users/w1850/.cache/huggingface"

# 2. 再 import
from sentence_transformers import SentenceTransformer

# 直接指向项目内的相对路径，禁用网络检查
embedding_model = SentenceTransformer(
    "models/text2vec-base-chinese",
    local_files_only=True
)

def embed_chunk(text):
    return embedding_model.encode(text).tolist()