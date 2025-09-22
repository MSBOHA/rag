"""
检索与召回逻辑接口和实现
"""


from typing import List, Optional

class BaseRetriever:
    def retrieve(self, query: str, top_k=5) -> List[str]:
        raise NotImplementedError

class Retriever(BaseRetriever):
    def __init__(self, vectordb, embedder, reranker: Optional[object] = None):
        self.vectordb = vectordb
        self.embedder = embedder
        self.reranker = reranker

    def retrieve(self, query: str, top_k=5, rerank_k: Optional[int] = None) -> List[dict]:
        # 1. 编码query
        query_vec = self.embedder.embed(query)
        # 2. 向量召回
        results = self.vectordb.search(query_vec, top_k=top_k if not rerank_k else rerank_k)
        # 3. 可选重排
        if self.reranker:
            texts = [r['metadata']['text'] for r in results]
            pairs = [(query, t) for t in texts]
            scores = self.reranker.predict(pairs)
            reranked = sorted(zip(results, scores), key=lambda x: x[1], reverse=True)
            results = [r for r, _ in reranked][:top_k]
        return results

def get_retriever(vectordb, embedder, reranker: Optional[object] = None) -> Retriever:
    return Retriever(vectordb, embedder, reranker)
