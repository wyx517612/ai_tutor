import chromadb
from typing import List
from sentence_transformers import CrossEncoder

chromadb_client=chromadb.EphemeralClient()
chromadb_collection=chromadb_client.get_or_create_collection(name="default")

def save_embeddings(chunks:list[str],embeddings:list[list[float]])->None:
    ids=[str(i) for i in range(len(chunks))]
    chromadb_collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=ids
    )
def retrieve(query:str,top_k:int)->list[str]:
    from embedder import embed_chunk
    query_embedding=embed_chunk(query)
    results=chromadb_collection.query(query_embeddings=[query_embedding],n_results=top_k)
    return results['documents'][0]

def rerank(query: str, retrieve_chunks: List[str], top_k: int) -> list[str]:
        cross_encoder = CrossEncoder('cross-encoder/mmarco-mMiniLMv2-L12-H384-v1')
        pairs = [(query, chunk) for chunk in retrieve_chunks]
        score = cross_encoder.predict(pairs)
        chunk_with_score_list = [(chunk, score) for chunk, score in zip(retrieve_chunks, score)]
        chunk_with_score_list.sort(key=lambda pair: pair[1], reverse=True)
        return [chunk for chunk, _ in chunk_with_score_list][:top_k]
