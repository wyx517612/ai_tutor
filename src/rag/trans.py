# src/rag/loader.py
from pypdf import PdfReader

def pdf_to_text(pdf_path: str) -> str:
    """将单个PDF文件转换为文本"""
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n\n"
    return text