from pypdf import PdfReader

# 把PDF转成txt
reader = PdfReader("rag/base_knowledge/data/2024.pdf")
text = ""
for page in reader.pages:
    page_text = page.extract_text()
    if page_text:
        text += page_text + "\n\n"

# 保存为txt
with open("rag/base_knowledge/data/2024.txt", "w", encoding="utf-8") as f:
    f.write(text)

print(f"✅ 转换完成，共 {len(text)} 字符")