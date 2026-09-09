from datetime import datetime

from langchain.tools import tool


@tool
def get_current_time():
    """当用户询问当前时间或日期时，调用此工具。"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")