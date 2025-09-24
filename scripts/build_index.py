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
    from src.llms.llm_api import get_llm
    from src.chains.rag_chain import RAGChain

    # 读取配置
    with open('configs/config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    doc_root = config.get('doc_path', None)
    split_method = config.get('split_method', 'paragraph')
    embedding_model = config.get('embedding_model', 'shibing624/text2vec-base-chinese')

    if not doc_root or not os.path.isdir(doc_root):
        raise ValueError('配置文件缺少 doc_path 或不是文件夹，请在 configs/config.yaml 中指定文档根目录路径')

    # 枚举所有子文件夹
    doc_folders = [d for d in os.listdir(doc_root) if os.path.isdir(os.path.join(doc_root, d))]
    if not doc_folders:
        raise ValueError(f'文档根目录 {doc_root} 下没有任何子文件夹')
    print("可用文档文件夹:")
    for i, dname in enumerate(doc_folders):
        print(f"[{i}] {dname}")
    idx = int(input("请选择要批量建库的文件夹编号: "))
    doc_dir = os.path.join(doc_root, doc_folders[idx])
    doc_folder_name = doc_folders[idx]

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
    print(f"[4/5] 初始化向量数据库，维度: {dim}")
    llm = get_llm(config.get('llm_model', 'gemini-2.5-flash'))

    # 按照文件夹名在indexes下创建子文件夹
    index_dir = os.path.join('indexes', doc_folder_name)
    os.makedirs(index_dir, exist_ok=True)

    # 只初始化一次库和RAGChain
    vectordb = get_vectordb(dim)
    retriever = get_retriever(vectordb, embedder)
    rag = RAGChain(None, splitter, embedder, vectordb, retriever, llm)
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
    save_path = os.path.join(index_dir, f"{doc_folder_name}_faiss.index")
    vectordb.save(save_path)
    print(f"✅ 向量库已保存到: {save_path}")

if __name__ == "__main__":
    main()
