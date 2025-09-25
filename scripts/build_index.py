import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
"""
构建/更新向量数据库索引脚本
"""

def main():
    import os
    import yaml
    from src.loaders.loader import auto_loader
    from src.splitters.splitter import get_splitter
    from src.embeddings.embedding import get_embedder
    from src.vectordb.vectordb import get_vectordb
    from src.retriever.retriever import get_retriever
    # 不需要 LLM 和 RAGChain
    # 读取配置
    with open('configs/config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    doc_dir = config.get('doc_path', None)
    split_method = config.get('split_method', 'paragraph')
    embedding_model = config.get('embedding_model', 'shibing624/text2vec-base-chinese')
    index_dir = config.get('index_path', None)

    if not doc_dir or not os.path.isdir(doc_dir):
        raise ValueError('配置文件缺少 doc_path 或不是文件夹，请在 configs/config.yaml 中指定文档根目录路径')
    if not index_dir:
        raise ValueError('配置文件缺少 index_path，请在 configs/config.yaml 中指定索引保存目录')
    os.makedirs(index_dir, exist_ok=True)

    # 枚举文件夹下所有文件
    file_list = [f for f in os.listdir(doc_dir) if os.path.isfile(os.path.join(doc_dir, f))]
    if not file_list:
        raise ValueError(f'文件夹 {doc_dir} 下没有可处理的文件')
    print(f"[1/5] 枚举文档: {file_list}")
    splitter_kwargs = {}
    if split_method == 'lines':
        lines_per_chunk = config.get('lines_per_chunk')
        if lines_per_chunk is None:
            raise ValueError('切分方式为lines时，必须在config.yaml中指定lines_per_chunk')
        splitter_kwargs['lines_per_chunk'] = lines_per_chunk
    elif split_method == 'length':
        max_length = config.get('max_length')
        overlap = config.get('overlap')
        if max_length is None or overlap is None:
            raise ValueError('切分方式为length时，必须在config.yaml中指定max_length和overlap')
        splitter_kwargs['max_length'] = max_length
        splitter_kwargs['overlap'] = overlap
    print(f"[2/5] 文本切分方式: {split_method}")
    splitter = get_splitter(method=split_method, **splitter_kwargs)
    print(f"[3/5] 加载Embedding模型: {embedding_model}")
    embedder = get_embedder(embedding_model)
    dim = len(embedder.embed("测试"))
    metric = config.get('metric', 'ip')
    index_type = config.get('index_type', 'flat')
    print(f"[4/5] 初始化向量数据库，维度: {dim}, metric: {metric}, index_type: {index_type}")
    vectordb = get_vectordb(dim, metric=metric, index_type=index_type)
    # BM25库（只要不是bm25就都建）
    from src.vectordb.vectordb import BM25VectorDB
    bm25db = BM25VectorDB()
    all_chunks = []
    all_metadatas = []
    for fname in file_list:
        file_path = os.path.join(doc_dir, fname)
        print(f"\n>>> 处理文件: {file_path}")
        loader = auto_loader(file_path)
        # 加载并切分
        docs = loader.load(file_path)
        if isinstance(docs, str):
            docs = [docs]
        for doc in docs:
            chunks = splitter.split(doc)
            all_chunks.extend(chunks)
            all_metadatas.extend([{"source": fname, "text": chunk} for chunk in chunks])
    # 批量嵌入并入库
    print(f"\n>>> 正在批量嵌入并写入向量库，总片段数: {len(all_chunks)}")
    embeddings = embedder.batch_embed(all_chunks) if hasattr(embedder, 'batch_embed') else [embedder.embed(x) for x in all_chunks]
    vectordb.add(embeddings, all_metadatas)
    bm25db.add(all_chunks, all_metadatas)
    # 保证每个库单独子文件夹
    subdir = os.path.join(index_dir, os.path.basename(os.path.normpath(doc_dir)))
    os.makedirs(subdir, exist_ok=True)
    save_path = os.path.join(subdir, f"{os.path.basename(os.path.normpath(doc_dir))}_faiss.index")
    vectordb.save(save_path)
    bm25_path = save_path.replace('_faiss.index', '_bm25.pkl')
    bm25db.save(bm25_path)
    print(f" 向量库已保存到: {save_path}")
    print(f" BM25库已保存到: {bm25_path}")

if __name__ == "__main__":
    main()
