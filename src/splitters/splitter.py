"""
文本/多模态切分器接口和实现
"""


from typing import List
import re

class BaseSplitter:
    def split(self, text: str) -> List[str]:
        raise NotImplementedError

class ParagraphSplitter(BaseSplitter):
    def split(self, text: str) -> List[str]:
        return [chunk.strip() for chunk in text.split("\n\n") if chunk.strip()]

class LineSplitter(BaseSplitter):
    def __init__(self, lines_per_chunk: int = 5):
        self.lines_per_chunk = lines_per_chunk
    def split(self, text: str) -> List[str]:
        lines = text.split('\n')
        return ['\n'.join(lines[i:i+self.lines_per_chunk]) for i in range(0, len(lines), self.lines_per_chunk)]

class LengthSplitter(BaseSplitter):
    def __init__(self, max_length: int = 500, overlap: int = 50):
        self.max_length = max_length
        self.overlap = overlap
    def split(self, text: str) -> List[str]:
        if len(text) <= self.max_length:
            return [text]
        chunks = []
        start = 0
        while start < len(text):
            end = start + self.max_length
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            start = max(start + 1, end - self.overlap) if end < len(text) else len(text)
        return chunks

def get_splitter(method: str = "paragraph", **kwargs) -> BaseSplitter:
    if method == "paragraph":
        return ParagraphSplitter()
    elif method == "lines":
        return LineSplitter(kwargs.get('lines_per_chunk', 5))
    elif method == "length":
        return LengthSplitter(kwargs.get('max_length', 500), kwargs.get('overlap', 50))
    else:
        raise ValueError(f"不支持的切分方式: {method}")
