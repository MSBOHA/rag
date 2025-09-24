"""
query rewrite + 意图分类模块
"""
from typing import List, Dict
from src.llms.llm_api import get_llm

def rewrite_and_classify_query(query: str, model_name: str = "qwen-plus") -> List[Dict]:
    """
    输入原始query，返回标准化表达和意图分类，支持多问题拆分。
    返回格式：[{'query': 标准化问题, 'intent': 意图类型}]
    """
    llm = get_llm("qwen-plus")  # 切分统一用 qwen-plus
    prompt = (
        "请将下列用户问题改写为标准表达，并判断其意图类型（事实/流程/定义/操作/闲聊）。"
        "如果包含多个问题，请拆分并分别输出。"
        "输出格式为JSON数组，每个元素包含 'query'（标准化问题）和 'intent'（意图类型）。"
        f"\n用户问题：{query}"
    )
    # 直接用llm.generate，假定llm能返回JSON字符串
    result = llm.generate(prompt, chunks=[])
    import json
    try:
        parsed = json.loads(result)
        if isinstance(parsed, list):
            return parsed[:3]  # 最多只保留前3个query
        else:
            return [{'query': query, 'intent': '未知'}]
    except Exception:
        return [{'query': query, 'intent': '未知'}]
