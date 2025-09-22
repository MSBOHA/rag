"""
RAG主流程链路
"""


from typing import List

class RAGChain:
    def __init__(self, loader, splitter, embedder, vectordb, retriever, llm):
        self.loader = loader
        self.splitter = splitter
        self.embedder = embedder
        self.vectordb = vectordb
        self.retriever = retriever
        self.llm = llm

    def build_index(self, doc_path, save_path=None) -> dict:
        """
        加载-切分-嵌入-入库，并可选保存
        返回chunks和embeddings，便于脚本保存元数据
        """
        docs = self.loader.load(doc_path)
        chunks = self.splitter.split(docs)
        embeddings = [self.embedder.embed(chunk) for chunk in chunks]
        self.vectordb.add(embeddings, metadatas=[{"text": c} for c in chunks])
        if save_path:
            self.vectordb.save(save_path)
        return {"chunks": chunks, "embeddings": embeddings}

    def query(self, query_text: str, top_k=5, rerank_k=None, return_chunks=True) -> dict:
        """
        检索-LLM生成，返回检索结果和答案
        """
        results = self.retriever.retrieve(query_text, top_k=top_k, rerank_k=rerank_k)
        if return_chunks:
            chunks = [r['metadata']['text'] for r in results]
        else:
            chunks = results
        answer = self.llm.generate(query_text, chunks)
        return {"results": results, "answer": answer}
