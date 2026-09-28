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
import logging
import time

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('api.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

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
    history: list = []

@app.get("/")
def read_root():
    return {"message": "考研助教API已启动", "chunks": len(all_chunks)}

@app.post("/chat")
def chat(request: ChatRequest):
    start_time = time.time()
    query = request.query
    history = request.history
    try:
    # 构建历史文本
        history_text = ""
        if history:
            for msg in history:
                role = "用户" if msg["role"] == "user" else "助教"
                history_text += f"{role}：{msg['content']}\n"
        agent_keywords = ["搜索", "最新", "2025", "2026", "大纲", "变化"]
        if any(kw in request.query for kw in agent_keywords):
            result = agent.invoke({"messages": [{"role": "user", "content": request.query}]})
            answer = result["messages"][-1].content
            elapsed = time.time() - start_time
            logger.info(f"query='{query[:20]}...' | source=agent | model=deepseek-chat | status=success | elapsed={elapsed:.2f}s")
            return {"answer": answer, "references": [], "source": "agent"}

        # 1. 检索
        retrieved = retrieve(request.query, top_k=5)
        # 2. 重排序
        reranked = rerank(request.query, retrieved, top_k=3)
        # 3. 生成回答
        answer,tokens= generator.generate_with_history(request.query, reranked,history_text)
        # 4. 返回
        elapsed = time.time() - start_time
        logger.info(f"query='{query[:20]}...' | source=rag | model=deepseek-chat | status=success | tokens={tokens} | elapsed={elapsed:.2f}s")
        return {
            "query": request.query,
            "answer": answer,
            "references": reranked,
            "source": "rag"
        }
    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"query='{query[:20]}...' | status=failed | error={str(e)} | elapsed={elapsed:.2f}s")
        return {"answer": "系统出错，请稍后重试", "references": []}



@app.post("/upload")
async def upload_files(files: list[UploadFile] = File(...)):
    start_time = time.time()
    try:
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
            elapsed = time.time() - start_time
            logger.info(f"upload | files={len(files)} | chunks={len(new_chunks)} | status=success | elapsed={elapsed:.2f}s")
            return {"message": f"成功上传 {len(files)} 个文件", "new_chunks": len(new_chunks)}
    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"upload | status=failed | error={str(e)} | elapsed={elapsed:.2f}s")
        return {"message": f"上传失败：{str(e)}", "new_chunks": 0}

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
    start_time = time.time()
    import random
    from openai import OpenAI
    try:
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
        elapsed = time.time() - start_time
        if start != -1 and end != 0:
            result = json.loads(content[start:end])

            logger.info(f"quiz | num_questions={request.num_questions} | model=deepseek-chat | status=success | tokens={response.usage.total_tokens} | elapsed={elapsed:.2f}s")
            return {"questions": result.get("questions", [])}
        else:
            logger.error(
                f"quiz | num_questions={request.num_questions} | "
                f"status=failed | error=JSON解析失败 | elapsed={elapsed:.2f}s"
            )
            return {"questions": [], "error": "解析题目失败"}
    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"quiz | status=failed | error={str(e)} | elapsed={elapsed:.2f}s")
        return {"questions": [], "error": f"生成题目失败：{str(e)}"}

