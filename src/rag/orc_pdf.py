import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io


# 如果Tesseract没在PATH里，取消下面这行的注释，改成你的安装路径
pytesseract.pytesseract.tesseract_cmd = r'D:\tesseract\tesseract.exe'

def ocr_pdf_to_text(pdf_path: str, dpi: int = 150) -> str:
    """
    对扫描件PDF进行OCR识别
    :param pdf_path: PDF文件路径
    :param dpi: 图片分辨率（越大越清晰，但速度更慢）
    :return: 识别的完整文本
    """
    full_text = ""
    doc = fitz.open(pdf_path)

    for page_num in range(len(doc)):
        print(f"📖 正在处理第 {page_num + 1} 页...")

        # 将PDF页面转为图片
        page = doc[page_num]
        # 设置缩放矩阵，提高清晰度
        zoom = dpi / 72
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)

        # 将图片数据转为PIL Image
        img_data = pix.tobytes("png")
        img = Image.open(io.BytesIO(img_data))

        # 使用Tesseract进行OCR识别
        # lang='chi_sim' 表示中文简体，'eng'表示英文
        page_text = pytesseract.image_to_string(img, lang='chi_sim+eng')

        # 如果识别结果为空，可能是页面格式问题，尝试只用英文识别
        if not page_text.strip():
            page_text = pytesseract.image_to_string(img, lang='eng')

        full_text += f"--- 第{page_num + 1}页 ---\n"
        full_text += page_text + "\n\n"

        print(f"  ✅ 第 {page_num + 1} 页识别完成，共 {len(page_text)} 个字符")

    doc.close()
    return full_text


if __name__ == "__main__":
    pdf_file = "rag/base_knowledge/data/09.pdf"  # 改成你的PDF路径

    print("🚀 开始OCR识别，可能需要几分钟...")
    try:
        result = ocr_pdf_to_text(pdf_file, dpi=300)

        # 保存结果
        output_path = "rag/base_knowledge/data/09_ocr_result.txt"
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(result)

        print(f"\n✅ OCR完成！共提取 {len(result)} 个字符")
        print(f"💾 结果已保存到：{output_path}")

        # 预览前300个字符
        print("\n📖 预览前300个字符：")
        print("-" * 40)
        print(result[:300])
        print("-" * 40)

    except FileNotFoundError:
        print(f"❌ 文件不存在: {pdf_file}")
    except Exception as e:
        print(f"❌ 发生错误: {e}")
        print("💡 请检查Tesseract是否已安装并配置正确")