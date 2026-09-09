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

# 3. 初始化模型
model = ChatOpenAI(model="deepseek-chat", temperature=0)

# 4. 创建提示模板
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个有用的助手，可以调用工具来回答问题"),
    ("user", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# 5. 创建Agent和执行器
tools = [get_current_time]
agent = create_openai_tools_agent(model, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# 6. 测试运行
if __name__ == "__main__":
    result = agent_executor.invoke({"input": "现在几点了？"})
    print("\n🧑‍🏫 最终回答:", result["output"])