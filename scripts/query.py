"""
查询脚本
"""

def main():
    import os
    import yaml
    from src.loaders.loader import auto_loader
    from src.splitters.splitter import get_splitter
    from src.embeddings.embedding import get_embedder
    from src.vectordb.vectordb import get_vectordb
    from src.retriever.retriever import get_retriever
    from src.llms.llm_api import get_llm
    from src.chains.rag_chain import RAGChain

    # 读取配置
    with open('configs/config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    db_dir = config.get('vector_db_path', None)
    doc_dir = config.get('doc_path', None)
    split_method = config.get('split_method', 'paragraph')
    embedding_model = config.get('embedding_model', 'shibing624/text2vec-base-chinese')
    rerank_model = config.get('rerank_model', None)
    llm_model = config.get('llm_model', 'gemini-2.5-flash')
    top_k = config.get('top_k', 5)
    rerank_k = config.get('rerank_k', 10)


    if not db_dir or not os.path.isdir(db_dir):
        raise ValueError('配置文件缺少 vector_db_path 或不是文件夹，请在 configs/config.yaml 中指定向量库文件夹路径')
    if not doc_dir or not os.path.isdir(doc_dir):
        raise ValueError('配置文件缺少 doc_path 或不是文件夹，请在 configs/config.yaml 中指定文档文件夹路径')

    # 选择要检索的库（文件夹名）
    db_folders = [d for d in os.listdir(db_dir) if os.path.isdir(os.path.join(db_dir, d))]
    if not db_folders:
        raise ValueError(f'向量库目录 {db_dir} 下没有任何子文件夹')
    print("可用向量库（文件夹）:")
    for i, dname in enumerate(db_folders):
        print(f"[{i}] {dname}")
    idx = int(input("请选择要检索的库编号: "))
    db_folder = db_folders[idx]
    db_folder_path = os.path.join(db_dir, db_folder)
    # 自动查找该文件夹下唯一的faiss.index文件
    db_files = [f for f in os.listdir(db_folder_path) if f.endswith('_faiss.index')]
    if not db_files:
        raise ValueError(f'库文件夹 {db_folder_path} 下没有faiss.index文件')
    if len(db_files) > 1:
        print("该库下有多个数据库文件，默认选第一个：", db_files[0])
    db_path = os.path.join(db_folder_path, db_files[0])
    print(f"[1/6] 加载向量数据库: {db_path}")
    loader = None  # 查询时不再需要loader
    print(f"[2/6] 文本切分方式: {split_method}")
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
    splitter = get_splitter(method=split_method, **splitter_kwargs)
    print(f"[3/6] 加载Embedding模型: {embedding_model}")
    embedder = get_embedder(embedding_model)
    dim = len(embedder.embed("测试"))
    print(f"[4/6] 加载向量数据库: {db_path}")
    vectordb = get_vectordb(dim, db_path)
    # 动态加载重排模型
    reranker = True
    if rerank_model:
        print(f"[5/6] 加载重排模型: {rerank_model}")
        try:
            from sentence_transformers import CrossEncoder
            reranker = CrossEncoder(rerank_model)
        except Exception as e:
            print(f"重排模型加载失败: {e}")
    else:
        print("[5/6] 未配置重排模型，跳过重排")
    retriever = get_retriever(vectordb, embedder, reranker)
    llm = get_llm(llm_model)

    print(f"[6/6] 检索并生成答案...")
    rag = RAGChain(None, splitter, embedder, vectordb, retriever, llm)
    query_text = input("请输入查询内容：")
    out = rag.query(query_text, top_k=top_k, rerank_k=rerank_k)
    print("检索结果：")
    for i, r in enumerate(out["results"]):
        print(f"[{i}] 分数: {r['score']:.4f}")
        print(f"内容: {r['metadata']['text'][:100]}...\n")
    print("\nLLM生成答案：")
    print(out["answer"])

if __name__ == "__main__":
    main()
