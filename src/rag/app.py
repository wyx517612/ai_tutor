import streamlit as st
from splitter import split_into_chunks
from embedder import embed_chunk
from retriever import save_embeddings, retrieve, rerank
from quiz_generator import Generator
import datetime
import os
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import tool
from pypdf import PdfReader


# 1. 定义工具（如果不需要时间工具，可以删除）
@tool
def get_current_time():
    """当用户询问当前日期或时间时，调用此工具"""
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# 2. 配置环境
os.environ["OPENAI_API_KEY"] = os.environ.get('DEEPSEEK_API_KEY')
os.environ["OPENAI_API_BASE"] = "https://api.deepseek.com/v1"

st.set_page_config(
    page_title="数据结构AI助教",
    page_icon="📚",
    layout="wide"
)

st.title("📚 数据结构AI助教")
st.caption("基于RAG（检索增强生成）技术，回答数据结构与算法相关问题")


@st.cache_resource
def init_rag():
    DOC_PATH = "rag/base_knowledge/data/2024.txt"
    chunks = split_into_chunks(DOC_PATH)
    embeddings = [embed_chunk(chunk) for chunk in chunks]
    save_embeddings(chunks, embeddings)
    generator = Generator()

    # 初始化 Agent（使用新版 API）
    model = ChatOpenAI(model="deepseek-chat", temperature=0)
    tools = [get_current_time]
    agent = create_agent(
        model=model,
        tools=tools,
        system_prompt="你是一个有用的助手，可以调用工具来回答问题。"
    )

    # 上传文件加载逻辑（从 uploads/ 文件夹加载）
    uploads_dir = "rag/base_knowledge/uploads"
    all_chunks = chunks.copy()  # 复制初始chunks
    if os.path.exists(uploads_dir):
        for filename in os.listdir(uploads_dir):
            if filename.endswith(".txt") or filename.endswith(".pdf"):
                file_path = os.path.join(uploads_dir, filename)
                try:
                    if filename.endswith(".txt"):
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                    else:  # pdf
                        from pypdf import PdfReader
                        reader = PdfReader(file_path)
                        content = ""
                        for page in reader.pages:
                            content += page.extract_text()

                    new_chunks = [chunk.strip() for chunk in content.split('\n\n') if chunk.strip()]
                    all_chunks.extend(new_chunks)
                    print(f"✅ 已加载上传文件: {filename}")
                except Exception as e:
                    print(f"⚠️ 加载 {filename} 失败: {e}")

    # 重新生成所有向量（包括上传的文件）
    embeddings = [embed_chunk(chunk) for chunk in all_chunks]
    save_embeddings(all_chunks, embeddings)
    generator = Generator()

    st.session_state.chunks = all_chunks

    from retriever import chromadb_collection
    actual_count = chromadb_collection.count()
    st.session_state.chunks = all_chunks  # 保留chunks用于检索
    st.session_state.doc_count = actual_count  # 新增：存储文档数

    # 在 return 之前，确保 chunks 被存入 session_state
    st.session_state.chunks = all_chunks
    st.session_state.doc_count = len(all_chunks)

    return all_chunks, generator, agent


with st.spinner("🧠 正在加载AI模型，请稍候..."):
    try:
        chunks, generator, agent = init_rag()
        st.success("✅ 系统已就绪！")
    except Exception as e:
        st.error(f"❌ 加载失败：{e}")
        st.stop()

with st.sidebar:
    st.header("📊 系统信息")
    # 从向量数据库直接读取文档数量
    from retriever import chromadb_collection

    actual_count = chromadb_collection.count()
    st.metric("知识库文档数", actual_count)

    st.metric("向量维度", 768)
    st.markdown("---")
    st.markdown("### 💡 示例问题")
    st.markdown("- 什么是栈？")
    st.markdown("- 解释一下二叉树")
    st.markdown("- 快速排序的时间复杂度")

    # ===== 上传资料区域 =====
    st.markdown("---")
    st.markdown("### 📤 上传你的资料")
    uploaded_files = st.file_uploader(
        "选择PDF或TXT文件（可多选）",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )

    if uploaded_files:
        if st.button("📚 加载到知识库", type="primary", use_container_width=True):
            with st.spinner(f"正在处理 {len(uploaded_files)} 个文件..."):
                # 确保 uploads 文件夹存在
                if not os.path.exists("rag/base_knowledge/uploads"):
                    os.makedirs("rag/base_knowledge/uploads")

                all_text = ""
                for uploaded_file in uploaded_files:
                    # 1. 保存文件到本地
                    file_path = os.path.join("rag/base_knowledge/uploads", uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                    # 2. 读取内容
                    if uploaded_file.type == "application/pdf":
                        reader = PdfReader(uploaded_file)
                        for page in reader.pages:
                            all_text += page.extract_text()
                    else:
                        all_text += uploaded_file.read().decode("utf-8")
                    all_text += "\n\n"

                # 3. 分割并加入向量库
                new_chunks = [chunk.strip() for chunk in all_text.split('\n\n') if chunk.strip()]
                new_embeddings = [embed_chunk(chunk) for chunk in new_chunks]
                save_embeddings(new_chunks, new_embeddings)

                # 4. 更新session_state
                if "chunks" not in st.session_state:
                    st.session_state.chunks = []
                st.session_state.chunks = st.session_state.chunks + new_chunks
                st.success(f"✅ 成功加载 {len(new_chunks)} 个新文本块！")
                st.cache_resource.clear()
                st.rerun()

query = st.text_input("✏️ 输入你的问题", placeholder="例如：什么是栈？")

# ===== 提问逻辑 =====
if st.button("🚀 提问", type="primary") and query:
    # 先判断是否包含时间关键词，走 Agent
    if "几点" in query or "时间" in query or "日期" in query:
        with st.spinner("⏰ 正在查询时间..."):
            result = agent.invoke({"messages": [{"role": "user", "content": query}]})
            answer = result["messages"][-1].content
        st.markdown("### 🧑‍🏫 助教回答")
        st.markdown(f"<div style='background-color:#f0f2f6;padding:20px;border-radius:10px;'>{answer}</div>",
                    unsafe_allow_html=True)
        st.caption("💡 该回答通过工具调用生成")
    else:
        # RAG 流程
        with st.spinner("🔍 正在检索知识库..."):
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