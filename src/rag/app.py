import streamlit as st
from splitter import split_into_chunks
from embedder import embed_chunk
from retriever import save_embeddings, retrieve, rerank
from quiz_generator import Generator
import datetime
import os
import json
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
    page_title="考研AI助教",
    page_icon="📚",
    layout="wide"
)

st.title("📚 考研AI助教")
st.caption("基于RAG（检索增强生成）技术，回答408考研相关问题")


@st.cache_resource
def init_rag():  # TODO: 后续支持多用户知识库隔离
    # ===== 1. 读取所有基础文档 =====
    base_dir = "base_knowledge/data"
    all_chunks = []
    if not os.path.exists(base_dir):
        print(f"⚠️ 目录不存在：{base_dir}")
    else:
        for filename in os.listdir(base_dir):
            if filename.endswith(".pdf"):
                file_path = os.path.join(base_dir, filename)
                print(f"📖 正在处理：{filename}")
                try:
                    reader = PdfReader(file_path)
                    text = ""
                    for page in reader.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n\n"
                    chunks = split_into_chunks(text, is_path=False)
                    all_chunks.extend(chunks)
                    print(f"   ✅ 已加载：{filename}，共 {len(chunks)} 个文本块")
                except Exception as e:
                    print(f"   ❌ 读取失败：{filename}，错误：{e}")
            elif filename.endswith(".txt"):
                file_path = os.path.join(base_dir, filename)
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                chunks = split_into_chunks(content, is_path=False)
                all_chunks.extend(chunks)
                print(f"✅ 已加载：{filename}，共 {len(chunks)} 个文本块")

    # ===== 2. 从磁盘恢复用户上传的chunks =====
    user_chunks_file = "base_knowledge/user_chunks.json"
    saved_user_chunks = []
    if os.path.exists(user_chunks_file):
        with open(user_chunks_file, "r", encoding="utf-8") as f:
            saved_user_chunks = json.load(f)
        print(f"✅ 从磁盘恢复 {len(saved_user_chunks)} 个用户上传的文本块")

    # ===== 3. 向量化并存入数据库（只在向量库为空时执行） =====
    from retriever import chromadb_collection
    if chromadb_collection.count() == 0:
        all_to_embed = all_chunks + saved_user_chunks
        embeddings = [embed_chunk(chunk) for chunk in all_to_embed]
        save_embeddings(all_to_embed, embeddings)
        print(f"✅ 已向量化 {len(all_to_embed)} 个文本块")
    else:
        print(f"✅ 向量库已有 {chromadb_collection.count()} 条数据，跳过向量化")

    generator = Generator()

    # 初始化 Agent
    model = ChatOpenAI(model="deepseek-chat", temperature=0)
    tools = [get_current_time]
    agent = create_agent(
        model=model,
        tools=tools,
        system_prompt="你是一个有用的助手，可以调用工具来回答问题。"
    )

    # 只在第一次初始化时设置 session_state
    if "chunks" not in st.session_state:
        st.session_state.chunks = all_chunks + saved_user_chunks
        st.session_state.base_chunks = all_chunks
        st.session_state.all_chunks = all_chunks + saved_user_chunks
        st.session_state.user_chunks = saved_user_chunks

    return all_chunks, saved_user_chunks, generator, agent


with st.spinner("🧠 正在加载AI模型，请稍候..."):
    try:
        base_chunks, saved_user_chunks, generator, agent = init_rag()
        st.success("✅ 系统已就绪！")
    except Exception as e:
        st.error(f"❌ 加载失败：{e}")
        st.stop()

# ===== 兜底：确保 session_state 始终有值（刷新后不丢） =====
if "base_chunks" not in st.session_state:
    st.session_state.base_chunks = base_chunks
    st.session_state.user_chunks = saved_user_chunks
    st.session_state.all_chunks = base_chunks + saved_user_chunks

with st.sidebar:
    st.header("📊 系统信息")
    st.metric("基础资料文档数", len(st.session_state.get("base_chunks", [])))
    st.metric("用户上传文档数", len(st.session_state.get("user_chunks", [])))
    st.metric("总计文档数", len(st.session_state.get("all_chunks", [])))
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
                if not os.path.exists("base_knowledge/uploads"):
                    os.makedirs("base_knowledge/uploads")

                all_text = ""
                for uploaded_file in uploaded_files:
                    # 1. 保存文件到本地
                    file_path = os.path.join("base_knowledge/uploads", uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                    # 2. 读取内容
                    if uploaded_file.type == "application/pdf":
                        reader = PdfReader(uploaded_file)
                        for page in reader.pages:
                            all_text += page.extract_text() + "\n\n"
                    else:
                        all_text += uploaded_file.read().decode("utf-8") + "\n\n"

                # 3. 分割并加入向量库
                new_chunks = [chunk.strip() for chunk in all_text.split('\n\n') if chunk.strip()]
                new_embeddings = [embed_chunk(chunk) for chunk in new_chunks]
                save_embeddings(new_chunks, new_embeddings)

                # 4. 更新 session_state
                st.session_state.chunks = st.session_state.get("chunks", []) + new_chunks
                st.session_state.user_chunks = st.session_state.get("user_chunks", []) + new_chunks
                st.session_state.all_chunks = st.session_state.get("all_chunks", []) + new_chunks

                # 5. 持久化到磁盘（关键：刷新后不丢）
                user_chunks_file = "base_knowledge/user_chunks.json"
                existing = []
                if os.path.exists(user_chunks_file):
                    with open(user_chunks_file, "r", encoding="utf-8") as f:
                        existing = json.load(f)
                existing.extend(new_chunks)
                with open(user_chunks_file, "w", encoding="utf-8") as f:
                    json.dump(existing, f, ensure_ascii=False)

                st.success(f"✅ 成功加载 {len(new_chunks)} 个新文本块！")
                st.rerun()  # ← 加上这一行，让页面重新运行，刷新统计数据

query = st.text_input("✏️ 输入你的问题", placeholder="例如：什么是栈？")

# ===== 提问逻辑 =====
if st.button("🚀 提问", type="primary") and query:
    if "几点" in query or "时间" in query or "日期" in query:
        with st.spinner("⏰ 正在查询时间..."):
            result = agent.invoke({"messages": [{"role": "user", "content": query}]})
            answer = result["messages"][-1].content
        st.markdown("### 🧑‍🏫 助教回答")
        st.markdown(f"<div style='background-color:#f0f2f6;padding:20px;border-radius:10px;'>{answer}</div>",
                    unsafe_allow_html=True)
        st.caption("💡 该回答通过工具调用生成")
    else:
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