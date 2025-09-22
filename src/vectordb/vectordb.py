"""
向量数据库接口和实现，支持多库管理和本地持久化
"""


import os
import numpy as np
import faiss
from typing import List, Optional

class BaseVectorDB:
    def add(self, vectors, metadatas=None):
        raise NotImplementedError
    def search(self, query_vector, top_k=5):
        raise NotImplementedError
    def save(self, path: str):
        raise NotImplementedError
    def load(self, path: str):
        raise NotImplementedError

class FaissVectorDB(BaseVectorDB):
    def __init__(self, dim: int, db_path: Optional[str] = None):
        self.dim = dim
        self.db_path = db_path
        self.index = faiss.IndexFlatIP(dim)
        self.metadatas = []
        if db_path and os.path.exists(db_path):
            self.load(db_path)

    def add(self, vectors: List[List[float]], metadatas: Optional[List[dict]] = None):
        arr = np.array(vectors, dtype=np.float32)
        self.index.add(arr)
        if metadatas:
            self.metadatas.extend(metadatas)
        else:
            self.metadatas.extend([None] * len(vectors))

    def search(self, query_vector: List[float], top_k=5):
        arr = np.array([query_vector], dtype=np.float32)
        scores, idxs = self.index.search(arr, top_k)
        results = []
        for i, idx in enumerate(idxs[0]):
            meta = self.metadatas[idx] if idx < len(self.metadatas) else None
            results.append({'score': float(scores[0][i]), 'index': int(idx), 'metadata': meta})
        return results

    def save(self, path: str):
        faiss.write_index(self.index, path)
        # 可选：保存metadatas到同名json
        import json
        if self.metadatas:
            with open(path + '.meta.json', 'w', encoding='utf-8') as f:
                json.dump(self.metadatas, f, ensure_ascii=False)

    def load(self, path: str):
        self.index = faiss.read_index(path)
        import json
        meta_path = path + '.meta.json'
        if os.path.exists(meta_path):
            with open(meta_path, 'r', encoding='utf-8') as f:
                self.metadatas = json.load(f)
        else:
            self.metadatas = []

def get_vectordb(dim: int, db_path: Optional[str] = None) -> FaissVectorDB:
    return FaissVectorDB(dim, db_path)
