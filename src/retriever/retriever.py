from typing import List, Optional, Any
from collections import OrderedDict

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
    

class HybridRetriever(BaseRetriever):
    def __init__(self, vectordb, embedder, bm25db=None, reranker=None, alpha=0.5):
        self.vectordb = vectordb
        self.embedder = embedder
        self.bm25db = bm25db
        self.reranker = reranker
        self.alpha = alpha

    def retrieve(self, query: str, top_k=5, rerank_k: Optional[int] = None) -> List[dict]:
        # 向量召回
        vec_results = []
        if self.vectordb:
            query_vec = self.embedder.embed(query)
            vec_results = self.vectordb.search(query_vec, top_k=top_k if not rerank_k else rerank_k)
        # BM25召回
        bm25_results = []
        if self.bm25db:
            bm25_results = self.bm25db.search(query, top_k=top_k if not rerank_k else rerank_k)
        # 合并去重
        all_results = OrderedDict()
        for r in vec_results:
            key = r['metadata']['text'] if r.get('metadata') and r['metadata'] else r.get('text')
            if key:
                all_results[key] = {'vec_score': r['score'], 'bm25_score': 0, 'metadata': r.get('metadata', {}), 'text': r.get('text', '')}
        for r in bm25_results:
            key = r['metadata']['text'] if r.get('metadata') and r['metadata'] else r.get('text')
            if key in all_results:
                all_results[key]['bm25_score'] = r['score']
            else:
                all_results[key] = {'vec_score': 0, 'bm25_score': r['score'], 'metadata': r.get('metadata', {}), 'text': r.get('text', '')}
        # RRF融合分数（Reciprocal Rank Fusion）
        # 计算每个结果在各自召回列表中的排名
        def get_rank(results, key_map):
            rank = {}
            for idx, r in enumerate(results):
                key = r['metadata']['text'] if r.get('metadata') and r['metadata'] else r.get('text')
                if key:
                    rank[key] = idx + 1  # 排名从1开始
            return rank

        vec_rank = get_rank(vec_results, all_results)
        bm25_rank = get_rank(bm25_results, all_results)
        k_rrf = 60  # RRF超参数，常用30~60
        merged = []
        for key, v in all_results.items():
            r1 = vec_rank.get(key, 10000)
            r2 = bm25_rank.get(key, 10000)
            rrf_score = 1/(k_rrf + r1) + 1/(k_rrf + r2)
            merged.append({'score': rrf_score, 'metadata': v['metadata'], 'text': v['text']})
        merged = sorted(merged, key=lambda x: x['score'], reverse=True)[:top_k]
        # 可选重排
        if self.reranker:
            texts = [m['text'] or m['metadata'].get('text', '') for m in merged]
            if hasattr(self.reranker, 'predict'):
                scores = self.reranker.predict([(query, t) for t in texts])
                for i, s in enumerate(scores):
                    merged[i]['score'] = float(s)
                merged = sorted(merged, key=lambda x: x['score'], reverse=True)
        return merged[:top_k]
    

def get_retriever(vectordb, embedder, reranker: Optional[object] = None, bm25db=None, alpha=0.5):
    if bm25db is not None and vectordb is not None:
        return HybridRetriever(vectordb, embedder, bm25db, reranker, alpha)
    return Retriever(vectordb, embedder, reranker)
