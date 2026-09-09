# app.py (放在 C:\Users\w1850\rag\src\rag\ 目录)
import streamlit as st
from agent_tools.test_agent import agent_executor
from splitter import split_into_chunks
from embedder import embed_chunk
from retriever import save_embeddings, retrieve, rerank
from quiz_generator import Generator
import datetime
import os
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import tool

# 1. 定义工具
@tool
def get_current_time():
    """当用户询问当前日期或时间时，调用此工具"""
    return datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")

# 2. 配置环境（请替换为你的真实密钥）
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
    DOC_PATH = "data/第一章 绪论.txt"
    chunks = split_into_chunks(DOC_PATH)
    embeddings = [embed_chunk(chunk) for chunk in chunks]
    save_embeddings(chunks, embeddings)
    generator = Generator()

# 3. 初始化模型
    model = ChatOpenAI(model="deepseek-chat", temperature=0)

# 4. 创建提示模板
    prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个有用的助手，可以调用工具来回答问题"),
    ("user", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])
# 5. 创建Agent和执行器
    tools = [get_current_time]
    agent = create_openai_tools_agent(model, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    return chunks, generator,agent_executor

with st.spinner("🧠 正在加载AI模型，请稍候..."):
    try:
        chunks, generator,agent_executor = init_rag()
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

if (st.button("🚀 提问", type="primary")) and query:
    if("几点" in query or "时间" in query or "日期" in query):
        with st.spinner("🔍 正在检索..."):
            result=agent_executor.invoke({"input":query})
            answer=result["output"]
    else:
        with st.spinner("🔍 正在检索..."):
            retrieved = retrieve(query, top_k=5)
            reranked = rerank(query, retrieved, top_k=3)
            answer = generator.generate(query, reranked)
        with st.expander("📖 查看参考资料"):
            for i, chunk in enumerate(reranked):
                st.markdown(f"**资料 {i + 1}**")
                st.write(chunk[:300] + "..." if len(chunk) > 300 else chunk)

    st.markdown("### 🧑‍🏫 助教回答")
    st.markdown(f"<div style='background-color:#black;padding:20px;border-radius:10px;'>{answer}</div>",
                unsafe_allow_html=True)

