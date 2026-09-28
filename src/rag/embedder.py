import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np

MODEL_PATH = r"C:\Users\w1850\rag\models\text2vec-base-chinese"

# use_fast=False 强制用 vocab.txt，绕开缺失的 tokenizer.json
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False, local_files_only=True)
model = AutoModel.from_pretrained(MODEL_PATH, local_files_only=True)
model.eval()

def embed_chunk(text: str):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    # 取 [CLS] 向量
    embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
    # 归一化
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm
    return embedding.tolist()