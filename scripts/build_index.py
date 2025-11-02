import sys
import os
import time
import yaml
import numpy as np

# 将项目根目录加入路径，便于模块导入
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.loaders.loader import auto_loader
from src.splitters.splitter import get_splitter
from src.embeddings.embedding import get_embedder
from src.vectordb.vectordb import get_vectordb, BM25VectorDB

"""
构建/更新向量数据库索引脚本
优化点：
- 采用与 embedding_test 类似的执行思路：预热 + 批量编码 + CUDA 同步，避免频繁 empty_cache
- 移除函数中部的临时导入，所有依赖在文件顶部导入
"""

def main():
    # 读取配置
    with open('configs/config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    time_start = time.time()
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
    precision = config.get('precision')  # 可选
    embedder = get_embedder(embedding_model, precision=precision)

    # 正确获取向量维度（单条输入返回 (1, dim)）
    test_emb = embedder.embed("测试", convert_to_tensor=False)
    dim = int(test_emb.shape[-1])
    # 可选：从配置设置最大序列长度，限制极长文本以提升稳定性
    max_seq_len_cfg = config.get('max_seq_length', None)
    if max_seq_len_cfg is not None:
        try:
            embedder.model.max_seq_length = int(max_seq_len_cfg)
            print(f"设置模型 max_seq_length={embedder.model.max_seq_length}")
        except Exception:
            pass
    metric = config.get('metric', 'ip')
    index_type = config.get('index_type', 'flat')
    print(f"[4/5] 初始化向量数据库，维度: {dim}, metric: {metric}, index_type: {index_type}")
    vectordb = get_vectordb(dim, metric=metric, index_type=index_type)
    # BM25 库（始终构建一份文本索引）
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


    print(f"\n>>> 正在批量嵌入并写入向量库，总片段数: {len(all_chunks)}")
    embedding_batch_size = int(config.get('embedding_batch_size', 32))
    per_request = int(config.get('embed_per_request', 32))

    total_enc_time = 0.0
    total_add_time = 0.0
    total_samples = 0

    for i in range(0, len(all_chunks), per_request):
        j = min(i + per_request, len(all_chunks))
        batch_chunks = all_chunks[i:j]
        batch_metas = all_metadatas[i:j]
        # 编码（简化，无自适应/同步/日志）
        t0 = time.perf_counter()
        emb_arr = embedder.embed(
            batch_chunks,
            batch_size=embedding_batch_size,
            normalize=False,
            convert_to_tensor=False,
        )
        enc_time = time.perf_counter() - t0
        total_enc_time += enc_time

        # 写入向量库
        t1 = time.perf_counter()
        vectordb.add(emb_arr, batch_metas)
        add_time = time.perf_counter() - t1
        total_add_time += add_time

        # BM25 追加
        bm25db.add(batch_chunks, batch_metas, rebuild=False)
        bm25_time = 0.0

        total_samples += len(batch_chunks)
        
        # 打印进度（可选）
        print(f"  处理片段 {i} - {j}，编码: {enc_time:.2f}s, 添加: {add_time:.2f}s, BM25: {bm25_time:.2f}s")

    # BM25 最终构建
    if getattr(bm25db, "_needs_build", False):
        tbb = time.time()
        print(">> 开始一次性构建 BM25 索引 ...")
        bm25db.build()
        print(">> BM25 build 用时:", time.time() - tbb)

    print(f"\n总编码耗时: {total_enc_time:.3f}s, 总写入耗时: {total_add_time:.3f}s, 总样本: {total_samples}")
    if total_enc_time > 0:
        print(f"平均编码吞吐: {total_samples/total_enc_time:.2f} samples/s")
    base = os.path.basename(os.path.normpath(doc_dir))
    save_path = os.path.join(index_dir, f"{base}_faiss.index")
    vectordb.save(save_path)
    bm25_path = save_path.replace('_faiss.index', '_bm25.pkl')
    bm25db.save(bm25_path)
    # 保存索引信息（便于查询端对齐嵌入模型与维度）
    try:
        import json
        info = {
            "embedding_model": embedding_model,
            "dim": dim,
            "metric": metric,
            "index_type": index_type,
            "max_seq_length": int(max_seq_len_cfg) if max_seq_len_cfg is not None else None,
            "precision": precision,
            "split_method": split_method,
            "doc_path": os.path.abspath(doc_dir),
            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        }
        with open(save_path + '.info.json', 'w', encoding='utf-8') as f:
            json.dump(info, f, ensure_ascii=False, indent=2)
    except Exception:
        pass
    print(f" 向量库已保存到: {save_path}")
    print(f" BM25库已保存到: {bm25_path}")
    time_end = time.time()
    print(f"总用时: {time_end - time_start} 秒")
if __name__ == "__main__":
    main()
