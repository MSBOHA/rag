"""
文档加载器接口和实现：支持txt、pdf、图片OCR等
"""


from typing import List
import os

class BaseLoader:
    def load(self, path: str) -> str:
        raise NotImplementedError

class TxtLoader(BaseLoader):
    def load(self, path: str) -> str:
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()

class PDFLoader(BaseLoader):
    def load(self, path: str) -> str:
        try:
            import fitz  # PyMuPDF
        except ImportError:
            raise ImportError('请先安装PyMuPDF: pip install pymupdf')
        doc = fitz.open(path)
        text = ''
        for page in doc:
            text += page.get_text()
        return text

class ImageOCRLoader(BaseLoader):
    def load(self, path: str) -> str:
        try:
            import pytesseract
            from PIL import Image
        except ImportError:
            raise ImportError('请先安装pytesseract和Pillow: pip install pytesseract pillow')
        image = Image.open(path)
        return pytesseract.image_to_string(image, lang='chi_sim')

def auto_loader(path: str) -> BaseLoader:
    ext = os.path.splitext(path)[-1].lower()
    if ext in ['.txt', '.md']:
        return TxtLoader()
    elif ext in ['.pdf']:
        return PDFLoader()
    elif ext in ['.png', '.jpg', '.jpeg', '.bmp']:
        return ImageOCRLoader()
    else:
        raise ValueError(f'不支持的文件类型: {ext}')
