"""
测试加载-切片-建库流程
"""
import os
from src.loaders.loader import auto_loader
from src.splitters.splitter import get_splitter
from src.embeddings.embedding import get_embedder
from src.vectordb.vectordb import get_vectordb

def test_load_split_embed_build():
    doc_path = os.path.join('data', 'doc.md')
    loader = auto_loader(doc_path)
    text = loader.load(doc_path)
    assert isinstance(text, str) and len(text) > 0
    print(text)

    splitter = get_splitter(method="paragraph")
    chunks = splitter.split(text)
    assert isinstance(chunks, list) and len(chunks) > 0
    print(f"切分块数: {len(chunks)}")
    embedder = get_embedder()
    embeddings = [embedder.embed(chunk) for chunk in chunks]
    assert len(embeddings) == len(chunks)
    print(f"嵌入向量数: {len(embeddings)}")
    print(f"嵌入向量维度: {len(embeddings[0])}")
    dim = len(embeddings[0])
    vectordb = get_vectordb(dim)
    vectordb.add(embeddings, metadatas=[{"text": c} for c in chunks])
    results = vectordb.search(embeddings[0], top_k=3)
    print(f"检索结果数: {len(results)}")
    print(results)
    assert isinstance(results, list) and len(results) > 0

    # 测试保存和加载
    save_path = os.path.join('indexes', 'test_faiss.index')
    vectordb.save(save_path)
    assert os.path.exists(save_path)

    # 重新加载并检索
    vectordb2 = get_vectordb(dim, save_path)
    results2 = vectordb2.search(embeddings[0], top_k=3)
    assert results2 == results
    print("加载-切片-建库-保存-加载-检索流程测试通过！")

if __name__ == "__main__":
    test_load_split_embed_build()
