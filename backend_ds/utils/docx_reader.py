from docx import Document

def read_docx(file_path):
    """读取 Word 文档(.docx)为纯文本"""
    try:
        doc = Document(file_path)
        full_text = []

        for para in doc.paragraphs:
            full_text.append(para.text)

        return '\n'.join(full_text)

    except Exception as e:
        print(f"读取docx失败: {e}")
        raise e