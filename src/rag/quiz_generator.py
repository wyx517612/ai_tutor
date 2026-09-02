# src/generator.py
import os
from openai import OpenAI
from typing import List


class Generator:
    def __init__(self, api_key: str = None, base_url: str = "https://api.deepseek.com"):
        # 如果没有传api_key，从环境变量读取
        if api_key is None:
            api_key = os.environ.get('DEEPSEEK_API_KEY')
            if api_key is None:
                raise ValueError("请设置 DEEPSEEK_API_KEY 环境变量或传入 api_key 参数")

        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        print("✅ DeepSeek生成器初始化完成")

    def generate(self, query: str, chunks: List[str]) -> str:
        """基于检索到的chunks生成回答"""
        context = "\n\n".join(chunks)

        prompt = f"""请根据以下参考资料回答用户问题。如果资料中没有相关信息，请明确告知用户。

参考资料：
{context}

用户问题：{query}

请基于参考资料给出准确且简洁的回答。"""

        try:
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "你是一个基于知识库回答问题的助手，只根据提供的参考资料回答问题。"},
                    {"role": "user", "content": prompt}
                ],
                stream=False,
                temperature=0.3,  # 降低随机性，让回答更准确
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"⚠️ 生成回答失败：{e}"