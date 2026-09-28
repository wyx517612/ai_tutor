import chromadb
from typing import List
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'  # 国内镜像
os.environ['HF_HUB_OFFLINE'] = '1'                   # 离线模式
os.environ['TRANSFORMERS_OFFLINE'] = '1'
from sentence_transformers import CrossEncoder
chromadb_client = chromadb.PersistentClient(path="./chroma_db")
chromadb_collection=chromadb_client.get_or_create_collection(name="default")

def save_embeddings(chunks:list[str],embeddings:list[list[float]])->None:
    ids=[str(i) for i in range(len(chunks))]
    chromadb_collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=ids
    )
def retrieve(query:str,top_k:int)->list[str]:
    print(f"🔍 当前向量库中的文档总数：{chromadb_collection.count()}")
    from embedder import embed_chunk
    query_embedding=embed_chunk(query)
    results=chromadb_collection.query(query_embeddings=[query_embedding],n_results=top_k)
    return results['documents'][0]

def rerank(query: str, retrieve_chunks: List[str], top_k: int) -> list[str]:
        cross_encoder = CrossEncoder("./models/cross-encoder",local_files_only=True)
        pairs = [(query, chunk) for chunk in retrieve_chunks]
        score = cross_encoder.predict(pairs)
        chunk_with_score_list = [(chunk, score) for chunk, score in zip(retrieve_chunks, score)]
        chunk_with_score_list.sort(key=lambda pair: pair[1], reverse=True)
        return [chunk for chunk, _ in chunk_with_score_list][:top_k]
