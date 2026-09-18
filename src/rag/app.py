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

# 1. 定义工具
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


@st.cache_resource #资源缓存 全局 streamlit点击时会重新执行代码 一直初始化很麻烦 所以缓存起来点击的时候不运行这个函数直接拿结果
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
                    reader = PdfReader(file_path)#只负责打开并解析成python对象——>可以访问其页面和元数据等
                    text = ""
                    for page in reader.pages:
                        page_text = page.extract_text()#把这页的文档转换成文字 extract取出
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

    # ===== 2. 从磁盘恢复用户上传的chunks =====#？？？？？？？？？？？？？？？？？
    user_chunks_file = "base_knowledge/user_chunks.json"#？？？？？？？？？？？？？
    saved_user_chunks = []
    if os.path.exists(user_chunks_file):
        with open(user_chunks_file, "r", encoding="utf-8") as f:
            saved_user_chunks = json.load(f)#？？？？？？？？？？？？？？？？？？？？？？
        print(f"✅ 从磁盘恢复 {len(saved_user_chunks)} 个用户上传的文本块")

    # ===== 3. 向量化并存入数据库（只在向量库为空时执行） =====
    from retriever import chromadb_collection
    if chromadb_collection.count() == 0:#？？？？？？？？？？？？？？？？？？？？
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

    # 只在第一次初始化时设置 session_state？？？？？？？？？？？？？
    if "chunks" not in st.session_state:#？？？？？？？？？？？？？？？
        st.session_state.chunks = all_chunks + saved_user_chunks
        st.session_state.base_chunks = all_chunks
        st.session_state.all_chunks = all_chunks + saved_user_chunks
        st.session_state.user_chunks = saved_user_chunks

    return all_chunks, saved_user_chunks, generator, agent


with st.spinner("🧠 正在加载AI模型，请稍候..."):
    try:
        base_chunks, saved_user_chunks, generator, agent = init_rag()#？？？？？？？？？？？？？
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
        label_visibility="collapsed"#隐藏label并移除它占用的空间
    )
    # ===== 自动出题区域 =====
    st.markdown("---")
    st.markdown("### 📝 自动出题")
    st.caption("基于当前知识库生成选择题")

    num_questions = st.slider("题目数量", min_value=1, max_value=5, value=3)

    if st.button("🎯 生成题目", type="primary", use_container_width=True):
        with st.spinner("🧠 正在生成题目..."):
            # 从当前所有chunks中随机抽取素材
            import random

            all_chunks = st.session_state.get("all_chunks", [])
            if not all_chunks:
                st.warning("⚠️ 知识库为空，请先上传资料")
            else:
                sample_size = min(num_questions * 3, len(all_chunks))
                selected = random.sample(all_chunks, sample_size)

                # 用你的Generator或直接调用DeepSeek生成题目
                # 这里我们复用generator的client
                from openai import OpenAI

                client = OpenAI(
                    api_key=os.environ.get('DEEPSEEK_API_KEY'),
                    base_url="https://api.deepseek.com"
                )

                context = "\n\n".join(selected)
                prompt = f"""你是一位408考研出题老师。请根据以下教材内容，生成{num_questions}道高质量选择题。

    教材内容：
    {context}

    要求：
    1. 每道题4个选项（A、B、C、D）
    2. 标注正确答案
    3. 给出详细解析
    4. 严格按照以下JSON格式输出（不要包含其他内容）：
    {{
        "questions": [
            {{
                "question": "题目内容",
                "options": ["A. 选项A", "B. 选项B", "C. 选项C", "D. 选项D"],
                "answer": "A",
                "explanation": "解析内容"
            }}
        ]
    }}
    """
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
                else:
                    st.error("❌ 解析题目失败")

    # 显示生成的题目
    if "questions" in st.session_state and st.session_state.questions:
        st.markdown("---")
        st.markdown("### 📝 题目列表")

        for i, q in enumerate(st.session_state.questions):
            if "error" in q:
                st.error(q["error"])
                continue

            with st.expander(f"题目 {i + 1}"):
                st.markdown(f"**{q.get('question', '')}**")
                for opt in q.get('options', []):
                    st.markdown(f"- {opt}")

                if st.button(f"🔍 查看答案 {i + 1}", key=f"show_answer_{i}"):
                    st.success(f"✅ 正确答案：**{q.get('answer', '')}**")
                    st.info(f"📖 解析：{q.get('explanation', '')}")

    if uploaded_files:
        if st.button("📚 加载到知识库", type="primary", use_container_width=True):#按钮类型首要次要 用容器宽度or字宽度
            with st.spinner(f"正在处理 {len(uploaded_files)} 个文件..."):                # 确保 uploads 文件夹存在
                if not os.path.exists("base_knowledge/uploads"):
                    os.makedirs("base_knowledge/uploads")

                all_text = ""
                for uploaded_file in uploaded_files:
                    # 1. 保存文件到本地
                    file_path = os.path.join("base_knowledge/uploads", uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())#返回字节

                    # 2. 读取内容
                    if uploaded_file.type == "application/pdf":#PDF 文件对应的 MIME 类型字符串
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

                # 5. 持久化到磁盘（关键：刷新后不丢）？？？？？？？？？？？？？？？？？？？？
                user_chunks_file = "base_knowledge/user_chunks.json"
                existing = []
                if os.path.exists(user_chunks_file):
                    with open(user_chunks_file, "r", encoding="utf-8") as f:
                        existing = json.load(f)#把josn内容转换成python列表
                existing.extend(new_chunks)
                with open(user_chunks_file, "w", encoding="utf-8") as f:
                    json.dump(existing, f, ensure_ascii=False)#把A写入B 确保中文原样不乱吗

                st.success(f"✅ 成功加载 {len(new_chunks)} 个新文本块！")
                st.rerun()  # ← 加上这一行，让页面重新运行，刷新统计数据  让侧边栏重新渲染

query = st.text_input("✏️ 输入你的问题", placeholder="例如：什么是栈？")

# ===== 提问逻辑 =====
if st.button("🚀 提问", type="primary") and query:
    if "几点" in query or "时间" in query or "日期" in query:
        with st.spinner("⏰ 正在查询时间..."):
            result = agent.invoke({"messages": [{"role": "user", "content": query}]})#messages是一个对话列表 可能有好几轮
            answer = result["messages"][-1].content#返回最后一轮的agent回答
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