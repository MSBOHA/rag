from rank_bm25 import BM25Okapi
import jieba
import pickle
"""
向量数据库接口和实现，支持多库管理和本地持久化
"""


import os
import numpy as np
import faiss
from typing import List, Optional

class BaseVectorDB:
    def __init__(self):
        pass
    def add(self, vectors, metadatas=None):
        raise NotImplementedError
    def search(self, query_vector, top_k=5):
        raise NotImplementedError
    def save(self, path: str):
        raise NotImplementedError
    def load(self, path: str):
        raise NotImplementedError
# BM25 检索实现
class BM25VectorDB(BaseVectorDB):
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path
        self.docs = []  # 原文
        self.metadatas = []
        self.corpus = []  # 分词后文档
        self.bm25 = None
        if db_path and os.path.exists(db_path):
            self.load(db_path)

    def add(self, docs: list, metadatas: Optional[list] = None):
        # docs: List[str]
        self.docs.extend(docs)
        self.corpus.extend([list(jieba.cut(doc)) for doc in docs])
        if metadatas:
            self.metadatas.extend(metadatas)
        else:
            self.metadatas.extend([None] * len(docs))
        self.bm25 = BM25Okapi(self.corpus)

    def search(self, query: str, top_k=5):
        if not self.bm25:
            self.bm25 = BM25Okapi(self.corpus)
        query_tokens = list(jieba.cut(query))
        scores = self.bm25.get_scores(query_tokens)
        top_idxs = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        results = []
        for i in top_idxs:
            results.append({
                'score': float(scores[i]),
                'index': i,
                'metadata': self.metadatas[i] if i < len(self.metadatas) else None,
                'text': self.docs[i]
            })
        return results

    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump({'docs': self.docs, 'metadatas': self.metadatas, 'corpus': self.corpus}, f)

    def load(self, path: str):
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.docs = data.get('docs', [])
            self.metadatas = data.get('metadatas', [])
            self.corpus = data.get('corpus', [])
            self.bm25 = BM25Okapi(self.corpus) if self.corpus else None

class FaissVectorDB(BaseVectorDB):
    def __init__(self, dim: int, db_path: Optional[str] = None, metric: str = 'ip', index_type: str = 'flat', nlist: int = 100, hnsw_m: int = 32):
        self.dim = dim
        self.db_path = db_path
        self.metric = metric
        self.index_type = index_type
        # 选择距离度量
        if metric == 'ip':
            index_cls = faiss.IndexFlatIP
        elif metric == 'l2':
            index_cls = faiss.IndexFlatL2
        elif metric == 'cosine':
            index_cls = faiss.IndexFlatIP  # 归一化后用内积实现余弦
        else:
            raise ValueError(f'不支持的距离度量: {metric}')
        # 选择索引类型
        if index_type == 'flat':
            self.index = index_cls(dim)
        elif index_type == 'ivf':
            quantizer = index_cls(dim)
            self.index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT if metric in ['ip', 'cosine'] else faiss.METRIC_L2)
            self._need_train = True
        elif index_type == 'hnsw':
            self.index = faiss.IndexHNSWFlat(dim, hnsw_m)
            self._need_train = False
        else:
            raise NotImplementedError(f'暂未实现索引类型: {index_type}')
        self.metadatas = []
        if db_path and os.path.exists(db_path):
            self.load(db_path)
        else:
            # IVF 索引需要训练
            if index_type == 'ivf':
                self._need_train = True
            else:
                self._need_train = False

    def add(self, vectors: List[List[float]], metadatas: Optional[List[dict]] = None):
        arr = np.array(vectors, dtype=np.float32)
        if self.metric == 'cosine':
            arr = arr / (np.linalg.norm(arr, axis=1, keepdims=True) + 1e-8)
        # IVF 索引需要训练
        if self.index_type == 'ivf' and self._need_train:
            self.index.train(arr)
            self._need_train = False
        self.index.add(arr)
        if metadatas:
            self.metadatas.extend(metadatas)
        else:
            self.metadatas.extend([None] * len(vectors))

    def search(self, query_vector: List[float], top_k=5):
        arr = np.array([query_vector], dtype=np.float32)
        if self.metric == 'cosine':
            arr = arr / (np.linalg.norm(arr, axis=1, keepdims=True) + 1e-8)
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

def get_vectordb(dim: int, db_path: Optional[str] = None, metric: str = 'ip', index_type: str = 'flat') -> FaissVectorDB:
    if index_type == 'bm25':
        return BM25VectorDB(db_path)
    return FaissVectorDB(dim, db_path, metric, index_type)
