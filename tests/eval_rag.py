"""
RAG系统自动化评测脚本
用法：python tests/eval_rag.py --mode retrieval --file tests/retrieval_eval.jsonl
    python tests/eval_rag.py --mode answer --file tests/answer_eval.jsonl
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
import argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np

def load_config():
    import yaml
    with open('configs/config.yaml', 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def recall_at_k(pred_chunks, gold_chunks, k=5):
    # pred_chunks: List[str], gold_chunks: List[str]
    for chunk in pred_chunks[:k]:
        for gold in gold_chunks:
            if gold.strip() in chunk:
                return 1
    return 0

def bleu_score(pred, ref):
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    smoothie = SmoothingFunction().method4
    def clean_text(text):
        import re
        # 去除markdown列表符号、换行、空格
        text = re.sub(r'[\*\-·•]+', '', text)
        text = re.sub(r'\n+', '', text)
        text = re.sub(r'\s+', '', text)
        return text
    pred = clean_text(pred)
    ref = clean_text(ref)
    if USE_JIEBA:
        import jieba
        ref_tokens = list(jieba.cut(ref))
        pred_tokens = list(jieba.cut(pred))
    else:
        ref_tokens = list(ref)
        pred_tokens = list(pred)
    return sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smoothie)

def rouge_l(pred, ref):
    import difflib
    def clean_text(text):
        import re
        text = re.sub(r'[\*\-·•]+', '', text)
        text = re.sub(r'\n+', '', text)
        text = re.sub(r'\s+', '', text)
        return text
    pred = clean_text(pred)
    ref = clean_text(ref)
    if USE_JIEBA:
        import jieba
        ref_tokens = list(jieba.cut(ref))
        pred_tokens = list(jieba.cut(pred))
    else:
        ref_tokens = list(ref)
        pred_tokens = list(pred)
    seq = difflib.SequenceMatcher(None, pred_tokens, ref_tokens)
    lcs = seq.find_longest_match(0, len(pred_tokens), 0, len(ref_tokens)).size
    recall = lcs / len(ref_tokens) if ref_tokens else 0
    precision = lcs / len(pred_tokens) if pred_tokens else 0
    f1 = 2 * recall * precision / (recall + precision + 1e-8)
    return f1

def main():
    import random
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['retrieval', 'answer'], required=True)
    parser.add_argument('--file', required=True)
    parser.add_argument('--top_k', type=int, default=5)
    parser.add_argument('--sample_k', type=int, default=5, help='随机抽样评测的样本数，默认5')
    parser.add_argument('--use_jieba', action='store_true', help='是否用jieba分词评测，默认不分词')
    args = parser.parse_args()
    config = load_config()
    # 初始化RAG组件
    from src.embeddings.embedding import get_embedder
    from src.vectordb.vectordb import get_vectordb
    from src.retriever.retriever import get_retriever
    from src.llms.llm_api import get_llm
    db_dir = config.get('vector_db_path') or config.get('index_path') or config.get('db_path')
    embedding_model = config.get('embedding_model', 'shibing624/text2vec-base-chinese')
    rerank_model = config.get('rerank_model', None)
    llm_model = config.get('llm_model', 'gemini-2.5-flash')
    metric = config.get('metric', 'ip')
    index_type = config.get('index_type', 'flat')
    # 自动查找向量库
    import os
    db_files = [f for f in os.listdir(db_dir) if f.endswith('_faiss.index')]
    db_path = os.path.join(db_dir, db_files[0])
    embedder = get_embedder(embedding_model)
    dim = len(embedder.embed("测试"))
    vectordb = get_vectordb(dim, db_path, metric=metric, index_type=index_type)
    from src.vectordb.vectordb import BM25VectorDB
    bm25_path = db_path.replace('_faiss.index', '_bm25.pkl')
    bm25db = BM25VectorDB(bm25_path) if os.path.exists(bm25_path) else None
    reranker = None
    if rerank_model:
        from sentence_transformers import CrossEncoder
        reranker = CrossEncoder(rerank_model)
    retriever = get_retriever(vectordb, embedder, reranker, bm25db=bm25db)
    llm = get_llm(llm_model)
    # 读取评测集
    with open(args.file, encoding='utf-8') as f:
        lines = [json.loads(l) for l in f if l.strip()]
    # 随机抽样k个样本
    if args.sample_k > 0 and args.sample_k < len(lines):
        lines = random.sample(lines, args.sample_k)
    from src.query_rewrite.rewrite import rewrite_and_classify_query
    global USE_JIEBA
    USE_JIEBA = args.use_jieba
    results = []
    mrrs = []
    use_rewrite = False  # 如需关闭改写，设为False
    if args.mode == 'retrieval':
        for item in tqdm(lines, desc='检索评测'):
            query = item['query']
            gold_chunks = item['relevant_chunks']
            if use_rewrite:
                rewrites = rewrite_and_classify_query(query)
                std_query = rewrites[0]['query'] if rewrites else query
            else:
                std_query = query
            retrieved = retriever.retrieve(std_query, top_k=args.top_k)
            pred_chunks = [r['metadata']['text'] if r.get('metadata') and r['metadata'] else r.get('text', '') for r in retrieved]
            # recall@k
            hit = recall_at_k(pred_chunks, gold_chunks, k=args.top_k)
            results.append(hit)
            # MRR
            rank = None
            for idx, chunk in enumerate(pred_chunks):
                if any(gold.strip() in chunk for gold in gold_chunks):
                    rank = idx + 1
                    break
            if rank:
                mrrs.append(1.0 / rank)
            else:
                mrrs.append(0.0)
        recall = np.mean(results)
        mrr = np.mean(mrrs)
        print(f"recall@{args.top_k}: {recall:.3f}  MRR@{args.top_k}: {mrr:.3f}")
    elif args.mode == 'answer':
        for idx, item in enumerate(tqdm(lines, desc='生成评测')):
            question = item['question']
            ref = item['reference_answer']
            if use_rewrite:
                rewrites = rewrite_and_classify_query(question)
                std_query = rewrites[0]['query'] if rewrites else question
            else:
                std_query = question
            retrieved = retriever.retrieve(std_query, top_k=args.top_k)
            chunks = [r['metadata']['text'] if r.get('metadata') and r['metadata'] else r.get('text', '') for r in retrieved]
            pred = llm.generate(std_query, chunks)
            bleu = bleu_score(pred, ref)
            rouge = rouge_l(pred, ref)
            print(f"\n=== 样本{idx+1} ===")
            print(f"问题: {question}")
            print(f"AI回答: {pred}")
            print(f"标准答案: {ref}")
            results.append({'bleu': bleu, 'rouge_l': rouge})
        bleu_avg = np.mean([r['bleu'] for r in results])
        rouge_avg = np.mean([r['rouge_l'] for r in results])
        print(f"BLEU: {bleu_avg:.3f}  ROUGE-L: {rouge_avg:.3f}")

if __name__ == '__main__':
    main()
