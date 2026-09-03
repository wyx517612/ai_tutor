# app.py (放在 C:\Users\w1850\rag\src\rag\ 目录)
import streamlit as st
import sys
import os
from splitter import split_into_chunks
from embedder import embed_chunk
from retriever import save_embeddings, retrieve, rerank
from quiz_generator import Generator

st.set_page_config(
    page_title="数据结构AI助教",
    page_icon="📚",
    layout="wide"
)

st.title("📚 数据结构AI助教")
st.caption("基于RAG（检索增强生成）技术，回答数据结构与算法相关问题")


@st.cache_resource
def init_rag():
    doc_path = ("data/第一章 绪论.txt")
    chunks = split_into_chunks(doc_path)
    embeddings = [embed_chunk(chunk) for chunk in chunks]
    save_embeddings(chunks, embeddings)
    generator = Generator()
    return chunks, generator


with st.spinner("🧠 正在加载AI模型，请稍候..."):
    try:
        chunks, generator = init_rag()
        st.success("✅ 系统已就绪！")
    except Exception as e:
        st.error(f"❌ 加载失败：{e}")
        st.stop()

with st.sidebar:
    st.header("📊 系统信息")
    st.metric("知识库文档数", len(chunks))
    st.metric("向量维度", 768)
    st.markdown("---")
    st.markdown("### 💡 示例问题")
    st.markdown("- 什么是栈？")
    st.markdown("- 解释一下二叉树")
    st.markdown("- 快速排序的时间复杂度")

query = st.text_input("✏️ 输入你的问题", placeholder="例如：什么是栈？")

if st.button("🚀 提问", type="primary") and query:
    with st.spinner("🔍 正在检索..."):
        retrieved = retrieve(query, top_k=5)
        reranked = rerank(query, retrieved, top_k=3)
        answer = generator.generate(query, reranked)

    st.markdown("### 🧑‍🏫 助教回答")
    st.markdown(f"<div style='background-color:#black;padding:20px;border-radius:10px;'>{answer}</div>",
                unsafe_allow_html=True)

    with st.expander("📖 查看参考资料"):
        for i, chunk in enumerate(reranked):
            st.markdown(f"**资料 {i + 1}**")
            st.write(chunk[:300] + "..." if len(chunk) > 300 else chunk)