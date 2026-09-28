import streamlit as st
import os
import json
import datetime
import requests
from pypdf import PdfReader
from splitter import split_into_chunks
from embedder import embed_chunk
from retriever import save_embeddings, retrieve, rerank
from quiz_generator import Generator
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.tools import tool
from openai import OpenAI

# ===== 页面配置 =====
st.set_page_config(
    page_title="考研AI助教",
    page_icon="📚",
    layout="wide"
)

st.title("📚 考研AI助教")
st.caption("基于RAG（检索增强生成）技术，回答408考研相关问题")

# ===== 配置环境 =====
os.environ["OPENAI_API_KEY"] = os.environ.get('DEEPSEEK_API_KEY')
os.environ["OPENAI_API_BASE"] = "https://api.deepseek.com/v1"


# ===== 定义工具 =====
@tool
def get_current_time():
    """当用户询问当前日期或时间时，调用此工具"""
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ===== 初始化RAG和Agent =====
@st.cache_resource
def init_all():
    # 1. 加载基础资料
    base_dir = "base_knowledge/data"
    all_chunks = []
    if os.path.exists(base_dir):
        for filename in os.listdir(base_dir):
            file_path = os.path.join(base_dir, filename)
            if filename.endswith(".pdf"):
                reader = PdfReader(file_path)
                text = ""
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n\n"
                chunks = split_into_chunks(text, is_path=False)
                all_chunks.extend(chunks)
            elif filename.endswith(".txt"):
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                chunks = split_into_chunks(content, is_path=False)
                all_chunks.extend(chunks)

    # 2. 恢复用户上传的chunks
    user_chunks_file = "base_knowledge/user_chunks.json"
    saved_user_chunks = []
    if os.path.exists(user_chunks_file):
        with open(user_chunks_file, "r", encoding="utf-8") as f:
            saved_user_chunks = json.load(f)

    # 3. 向量化
    from retriever import chromadb_collection
    if chromadb_collection.count() == 0:
        all_to_embed = all_chunks + saved_user_chunks
        embeddings = [embed_chunk(chunk) for chunk in all_to_embed]
        save_embeddings(all_to_embed, embeddings)

    # 4. 初始化生成器
    generator = Generator()

    # 5. 初始化Agent（带Tavily搜索）
    from langchain_community.tools import TavilySearchResults
    model = ChatOpenAI(model="deepseek-chat", temperature=0)
    search_tool = TavilySearchResults(
        tavily_api_key=os.environ.get('TAVILY_API_KEY'),
        max_results=3
    )
    tools = [get_current_time, search_tool]
    agent = create_agent(
        model=model,
        tools=tools,
        system_prompt="你是一个考研助教，可以调用工具搜索最新信息。"
    )

    return all_chunks, saved_user_chunks, generator, agent


# ===== 加载 =====
with st.spinner("🧠 正在加载AI模型，请稍候..."):
    try:
        base_chunks, saved_user_chunks, generator, agent = init_all()
        st.success("✅ 系统已就绪！")
    except Exception as e:
        st.error(f"❌ 加载失败：{e}")
        st.stop()

# ===== 侧边栏 =====
with st.sidebar:
    st.header("📊 系统信息")
    st.metric("基础资料文档数", len(base_chunks))
    st.metric("用户上传文档数", len(saved_user_chunks))
    st.metric("总计文档数", len(base_chunks) + len(saved_user_chunks))

    st.markdown("---")
    st.markdown("### 💡 示例问题")
    st.markdown("- 什么是栈？")
    st.markdown("- 解释一下二叉树")
    st.markdown("- 2026年408大纲有什么变化？")

    # 上传资料
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
                os.makedirs("base_knowledge/uploads", exist_ok=True)
                all_text = ""
                for uploaded_file in uploaded_files:
                    file_path = os.path.join("base_knowledge/uploads", uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                    if uploaded_file.type == "application/pdf":
                        reader = PdfReader(uploaded_file)
                        for page in reader.pages:
                            all_text += page.extract_text() + "\n\n"
                    else:
                        all_text += uploaded_file.read().decode("utf-8") + "\n\n"

                new_chunks = [chunk.strip() for chunk in all_text.split('\n\n') if chunk.strip()]
                new_embeddings = [embed_chunk(chunk) for chunk in new_chunks]
                save_embeddings(new_chunks, new_embeddings)

                user_chunks_file = "base_knowledge/user_chunks.json"
                existing = []
                if os.path.exists(user_chunks_file):
                    with open(user_chunks_file, "r", encoding="utf-8") as f:
                        existing = json.load(f)
                existing.extend(new_chunks)
                with open(user_chunks_file, "w", encoding="utf-8") as f:
                    json.dump(existing, f, ensure_ascii=False)

                st.success(f"✅ 成功加载 {len(new_chunks)} 个新文本块！")
                st.rerun()

    # 自动出题
    st.markdown("---")
    st.markdown("### 📝 自动出题")
    num_questions = st.slider("题目数量", min_value=1, max_value=5, value=3)

    if st.button("🎯 生成题目", type="primary", use_container_width=True):
        with st.spinner("🧠 正在生成题目..."):
            import random

            all_chunks = base_chunks + saved_user_chunks
            if not all_chunks:
                st.warning("⚠️ 知识库为空，请先上传资料")
            else:
                sample_size = min(num_questions * 3, len(all_chunks))
                selected = random.sample(all_chunks, sample_size)
                client = OpenAI(
                    api_key=os.environ.get('DEEPSEEK_API_KEY'),
                    base_url="https://api.deepseek.com"
                )
                context = "\n\n".join(selected)
                prompt = f"""你是一位408考研出题老师。请根据以下教材内容，生成{num_questions}道高质量选择题。
教材内容：{context}
要求：4个选项，标注正确答案，给出解析，严格JSON格式输出。"""
                response = client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[
                        {"role": "system", "content": "你是一位408考研出题专家，只输出JSON格式。"},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7
                )
                content = response.choices[0].message.content
                start = content.find('{')
                end = content.rfind('}') + 1
                if start != -1 and end != 0:
                    result = json.loads(content[start:end])
                    st.session_state.questions = result.get('questions', [])

    if "questions" in st.session_state and st.session_state.questions:
        for i, q in enumerate(st.session_state.questions):
            with st.expander(f"题目 {i + 1}"):
                st.markdown(f"**{q.get('question', '')}**")
                for opt in q.get('options', []):
                    st.markdown(f"- {opt}")
                if st.button(f"🔍 查看答案 {i + 1}", key=f"show_answer_{i}"):
                    st.success(f"✅ 正确答案：**{q.get('answer', '')}**")
                    st.info(f"📖 解析：{q.get('explanation', '')}")

# ===== 对话历史 =====
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ===== 提问逻辑 =====
query = st.text_input("✏️ 输入你的问题", placeholder="例如：什么是栈？")

if st.button("🚀 提问", type="primary") and query:
    with st.spinner("🔍 正在检索知识库..."):
        agent_keywords = ["搜索", "最新", "2025", "2026", "大纲", "变化", "几点", "时间", "日期"]
        if any(kw in query for kw in agent_keywords):
            result = agent.invoke({"messages": [{"role": "user", "content": query}]})
            answer = result["messages"][-1].content
            reranked = []
        else:
            retrieved = retrieve(query, top_k=5)
            reranked = rerank(query, retrieved, top_k=3)
            answer = generator.generate(query, reranked)

        st.session_state.chat_history.append({"role": "user", "content": query})
        st.session_state.chat_history.append({"role": "assistant", "content": answer})

    st.markdown("### 🧑‍🏫 助教回答")
    st.markdown(f"<div style='background-color:#f0f2f6;padding:20px;border-radius:10px;'>{answer}</div>",
                unsafe_allow_html=True)

    if reranked:
        with st.expander("📖 查看参考资料"):
            for i, chunk in enumerate(reranked):
                st.markdown(f"**资料 {i + 1}**")
                st.write(chunk[:300] + "..." if len(chunk) > 300 else chunk)