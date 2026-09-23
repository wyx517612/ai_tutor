from fastapi import FastAPI
from pydantic import BaseModel
from splitter import split_into_chunks
from embedder import embed_chunk
from langchain_openai import ChatOpenAI
from fastapi import UploadFile, File
import shutil
from pydantic import BaseModel
import streamlit as st
from splitter import split_into_chunks
from embedder import embed_chunk
from retriever import save_embeddings, retrieve, rerank
from quiz_generator import Generator
import datetime
import os
import json
from langchain.agents import create_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import tool
from pypdf import PdfReader
import requests
from langchain_community.tools import TavilySearchResults

# 配置DeepSeek
os.environ["OPENAI_API_KEY"] = os.environ.get('DEEPSEEK_API_KEY')
os.environ["OPENAI_API_BASE"] = "https://api.deepseek.com/v1"

app = FastAPI(title="考研助教API")

# 初始化RAG（启动时执行一次）
print("🧠 正在初始化RAG...")
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
print("✅ RAG初始化完成")

# 定义联网搜索工具
search_tool = TavilySearchResults(
    tavily_api_key="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",
    max_results=3
)
# 初始化 Agent
model = ChatOpenAI(model="deepseek-chat", temperature=0)
tools = [search_tool]
agent = create_agent(
    model=model,
    tools=tools,
    system_prompt="你是一个考研助教，可以调用工具搜索最新消息。"
)



# 定义请求体
class ChatRequest(BaseModel):
    query: str

@app.get("/")
def read_root():
    return {"message": "考研助教API已启动", "chunks": len(all_chunks)}

@app.post("/chat")
def chat(request: ChatRequest):
    agent_keywords = ["搜索", "最新", "2025", "2026", "大纲", "变化"]
    if any(kw in request.query for kw in agent_keywords):
        result = agent.invoke({"messages": [{"role": "user", "content": request.query}]})
        answer = result["messages"][-1].content
        return {"answer": answer, "references": [], "source": "agent"}

    # 1. 检索
    retrieved = retrieve(request.query, top_k=5)
    # 2. 重排序
    reranked = rerank(request.query, retrieved, top_k=3)
    # 3. 生成回答
    answer = generator.generate(request.query, reranked)
    # 4. 返回
    return {
        "query": request.query,
        "answer": answer,
        "references": reranked
    }


@app.post("/upload")
async def upload_files(files: list[UploadFile] = File(...)):
    upload_dir = "base_knowledge/uploads"
    os.makedirs(upload_dir, exist_ok=True)

    all_text = ""
    for file in files:
        file_path = os.path.join(upload_dir, file.filename)
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        if file.filename.endswith(".pdf"):
            reader = PdfReader(file_path)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    all_text += page_text + "\n\n"
        else:
            with open(file_path, "r", encoding="utf-8") as f:
                all_text += f.read() + "\n\n"

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

    return {"message": f"成功上传 {len(files)} 个文件", "new_chunks": len(new_chunks)}

@app.get("/stats")
def get_stats():
    from retriever import chromadb_collection
    return {
        "base_chunks": len(all_chunks),
        "user_chunks": len(saved_user_chunks),
        "total_chunks": chromadb_collection.count()
    }





class QuizRequest(BaseModel):
    num_questions: int = 3


@app.post("/quiz")
def generate_quiz(request: QuizRequest):
    import random
    from openai import OpenAI

    all_knowledge = all_chunks + saved_user_chunks
    if not all_knowledge:
        return {"questions": [], "error": "知识库为空"}

    sample_size = min(request.num_questions * 3, len(all_knowledge))
    selected = random.sample(all_knowledge, sample_size)

    client = OpenAI(
        api_key=os.environ.get('DEEPSEEK_API_KEY'),
        base_url="https://api.deepseek.com"
    )

    context = "\n\n".join(selected)
    prompt = f"""你是一位408考研出题老师。请根据以下教材内容，生成{request.num_questions}道高质量选择题。

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
        return {"questions": result.get("questions", [])}
    else:
        return {"questions": [], "error": "解析题目失败"}