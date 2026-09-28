
import os

# 1. 先设置缓存目录（必须在任何相关 import 之前）
os.environ["HF_HOME"] = "C:/Users/w1850/.cache/huggingface"  # 换回你原来的缓存路径
os.environ["TRANSFORMERS_CACHE"] = "C:/Users/w1850/.cache/huggingface"
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 2. 再 import
from sentence_transformers import SentenceTransformer

# 3. 加载时明确告诉它：只用本地文件，不联网检查
embedding_model = SentenceTransformer(
    "shibing624/text2vec-base-chinese",
    local_files_only=True  # 关键参数
)
def embed_chunk(text):
    """把文本转换成向量"""
    return embedding_model.encode(text)