"""
查询脚本
"""

def main():
    import os
    import yaml
    import sys
    from src.embeddings.embedding import get_embedder
    from src.vectordb.vectordb import get_vectordb
    from src.retriever.retriever import get_retriever
    from src.llms.llm_api import get_llm

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
    print(f"[1/5] 加载向量数据库: {db_path}")
    loader = None  # 查询时不再需要loader
    # 不再需要文本切分方式，直接检索
    print(f"[2/5] 加载Embedding模型: {embedding_model}")
    embedder = get_embedder(embedding_model)
    dim = len(embedder.embed("测试"))
    print(f"[3/5] 加载向量数据库: {db_path}")
    vectordb = get_vectordb(dim, db_path)
    # 动态加载重排模型
    reranker = True
    if rerank_model:
        print(f"[4/5] 加载重排模型: {rerank_model}")
        try:
            from sentence_transformers import CrossEncoder
            reranker = CrossEncoder(rerank_model)
        except Exception as e:
            print(f"重排模型加载失败: {e}")
    else:
        print("[4/5] 未配置重排模型，跳过重排")
    retriever = get_retriever(vectordb, embedder, reranker)
    llm = get_llm(llm_model)

    is_chat = '--chat' in sys.argv
    if is_chat:
        print("多轮对话模式已开启，输入 exit/quit 结束。")
        messages = []
        while True:
            query_text = input("用户: ")
            if query_text.strip().lower() in ("exit", "quit"): break
            results = retriever.retrieve(query_text, top_k=top_k, rerank_k=rerank_k)
            print("检索结果：")
            for i, r in enumerate(results):
                print(f"[{i}] 分数: {r['score']:.4f}")
                print(f"内容: {r['metadata']['text'][:100]}...\n")
            chunks = [r['metadata']['text'] for r in results]
            # 多轮对话，历史messages+当前问题
            messages.append({"role": "user", "content": query_text})
            answer = llm.generate(query_text, chunks, messages=messages)
            print("LLM: ", answer)
            messages.append({"role": "assistant", "content": answer})
    else:
        print(f"[5/5] 检索并生成答案...")
        query_text = input("请输入查询内容：")
        results = retriever.retrieve(query_text, top_k=top_k, rerank_k=rerank_k)
        print("检索结果：")
        for i, r in enumerate(results):
            print(f"[{i}] 分数: {r['score']:.4f}")
            print(f"内容: {r['metadata']['text'][:100]}...\n")
        chunks = [r['metadata']['text'] for r in results]
        answer = llm.generate(query_text, chunks)
        print("\nLLM生成答案：")
        print(answer)

if __name__ == "__main__":
    main()
