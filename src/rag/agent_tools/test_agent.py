import os
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import tool
from datetime import datetime

# 1. 定义工具
@tool
def get_current_time():
    """当用户询问当前时间或日期时，调用此工具。"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# 2. 配置环境
os.environ["OPENAI_API_KEY"] = "你的DeepSeek API Key"
os.environ["OPENAI_API_BASE"] = "https://api.deepseek.com/v1"

# 3. 初始化模型
model = ChatOpenAI(model="deepseek-chat", temperature=0)

# 4. 创建 Agent（使用新版 API）
tools = [get_current_time]
agent = create_agent(
    model=model,
    tools=tools,
    system_prompt="你是一个有用的助手，可以调用工具来回答问题。"
)

# 5. 测试运行
if __name__ == "__main__":
    result = agent.invoke({"messages": [{"role": "user", "content": "现在几点了？"}]})
    print("\n🧑‍🏫 最终回答:", result["messages"][-1].content)