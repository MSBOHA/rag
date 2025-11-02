import os
import yaml
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import time
def load_config():
    with open('configs/config.yaml', 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def main():
    from src.embeddings.embedding import get_embedder
    from src.vectordb.vectordb import get_vectordb
    from src.retriever.retriever import get_retriever
    from src.llms.llm_api import get_llm
    start_time = time.time()
    config = load_config()
    db_dir = config.get('vector_db_path') or config.get('index_path') or config.get('db_path')
    if not db_dir or not os.path.isdir(db_dir):
        raise ValueError('配置文件缺少 vector_db_path/index_path/db_path 或不是文件夹，请在 configs/config.yaml 中指定向量库文件夹路径')

    # 自动查找该文件夹下唯一的 *_faiss.index 文件
    db_files = [f for f in os.listdir(db_dir) if f.endswith('_faiss.index')]
    if not db_files:
        raise ValueError(f'向量库文件夹 {db_dir} 下没有 *_faiss.index 文件')
    if len(db_files) > 1:
        print(f"该文件夹下有多个数据库文件，默认选第一个：{db_files[0]}")
    db_path = os.path.join(db_dir, db_files[0])
    print(f"[1/5] 读取向量数据库路径: {db_path}")

    embedding_model = config.get('embedding_model', 'shibing624/text2vec-base-chinese')
    rerank_model = config.get('rerank_model', None)
    llm_model = config.get('llm_model', 'gemini-2.5-flash')
    top_k = config.get('top_k', 5)
    rerank_k = config.get('rerank_k', 10)

    # 如存在索引信息文件，则优先使用其中记录的嵌入模型与参数，避免维度不一致
    index_info = None
    info_path = db_path + '.info.json'
    if os.path.exists(info_path):
        try:
            import json
            with open(info_path, 'r', encoding='utf-8') as f:
                index_info = json.load(f)
        except Exception:
            index_info = None

    precision = config.get('precision')
    max_seq_len_cfg = config.get('max_seq_length', None)
    if index_info:
        # 覆盖到与索引一致的 embed 配置（以减少出错概率）
        if index_info.get('embedding_model'):
            embedding_model = index_info['embedding_model']
        if index_info.get('precision'):
            precision = index_info.get('precision')
        if index_info.get('max_seq_length') is not None and max_seq_len_cfg is None:
            max_seq_len_cfg = index_info.get('max_seq_length')
    print(f"[2/5] 加载Embedding模型: {embedding_model}")
    embedder = get_embedder(embedding_model, precision=precision)
    # 设置 max_seq_length（如配置）
    if max_seq_len_cfg is not None:
        try:
            embedder.model.max_seq_length = int(max_seq_len_cfg)
        except Exception:
            pass
    # 正确获取向量维度
    test_emb = embedder.embed("测试", convert_to_tensor=False)
    dim = int(test_emb.shape[-1])
    metric = config.get('metric', 'ip')
    index_type = config.get('index_type', 'flat')
    print(f"[3/5] 加载向量数据库: {db_path}, metric: {metric}, index_type: {index_type}")
    vectordb = get_vectordb(dim, db_path, metric=metric, index_type=index_type)
    # 维度一致性校验（embedder vs index）
    if hasattr(vectordb, 'dim') and vectordb.dim is not None:
        index_dim = int(vectordb.dim)
        if dim != index_dim:
            raise ValueError(
                f"Embedding 维度({dim}) 与索引维度({index_dim})不一致。\n"
                f"- 当前嵌入模型: {embedding_model}\n"
                f"- 索引文件: {db_path}\n"
                f"请在 configs/config.yaml 中将 embedding_model 调整为构建索引时使用的模型，或重新用当前模型重建索引。"
            )
    # 尝试加载BM25库
    from src.vectordb.vectordb import BM25VectorDB
    bm25_path = db_path.replace('_faiss.index', '_bm25.pkl')
    bm25db = None
    if os.path.exists(bm25_path):
        bm25db = BM25VectorDB(bm25_path)
        print(f"[3.5/5] 加载BM25库: {bm25_path}")
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
    retriever = get_retriever(vectordb, embedder, reranker, bm25db=bm25db)
    llm = get_llm(llm_model)
    end_time = time.time()
    print(f"初始化完成，耗时: {end_time - start_time:.2f} 秒")
    # 支持 --chat 多轮对话模式
    from src.query_rewrite.rewrite import rewrite_and_classify_query
    is_chat = '--chat' in sys.argv
    if is_chat:
        print("多轮对话模式已开启，输入 exit/quit 结束。")
        messages = []
        while True:
            query_text = input("用户: ")
            if query_text.strip().lower() in ("exit", "quit"): break
            # 改写并分类
            rewrites = rewrite_and_classify_query(query_text, llm_model)
            print(rewrites)
            for item in rewrites:
                std_query = item['query']
                intent = item['intent']
                print(f"[Query Rewrite] 标准化: {std_query} | 意图: {intent}")
                results = retriever.retrieve(std_query, top_k=top_k, rerank_k=rerank_k)
                print("检索结果：")
                for i, r in enumerate(results):
                    print(f"[{i}] 分数: {r['score']:.4f}")
                    print(f"内容: {r['metadata']['text'][:100]}...\n")
                chunks = [r['metadata']['text'] for r in results]
                messages.append({"role": "user", "content": std_query})
                answer = llm.generate(std_query, chunks, messages=messages)
                print("LLM: ", answer)
                messages.append({"role": "assistant", "content": answer})
    else:
        print(f"[5/5] 检索并生成答案...")
        query_text = input("请输入查询内容：")
        rewrites = rewrite_and_classify_query(query_text, llm_model)
        for item in rewrites:
            std_query = item['query']
            intent = item['intent']
            print(f"[Query Rewrite] 标准化: {std_query} | 意图: {intent}")
            results = retriever.retrieve(std_query, top_k=top_k, rerank_k=rerank_k)
            print("检索结果：")
            for i, r in enumerate(results):
                print(f"[{i}] 分数: {r['score']:.4f}")
                print(f"内容: {r['metadata']['text'][:100]}...\n")
            chunks = [r['metadata']['text'] for r in results]
            answer = llm.generate(std_query, chunks)
            print("\nLLM生成答案：")
            print(answer)

if __name__ == "__main__":
    main()
