"""
嵌入生成接口和实现
"""


from typing import List

class BaseEmbedding:
    def embed(self, text: str) -> List[float]:
        raise NotImplementedError

class SentenceTransformerEmbedding(BaseEmbedding):
    def __init__(self, model_name: str = "shibing624/text2vec-base-chinese"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
    def embed(self, text: str) -> List[float]:
        return self.model.encode(text, normalize_embeddings=True).tolist()

def get_embedder(model_name: str = "shibing624/text2vec-base-chinese") -> BaseEmbedding:
    return SentenceTransformerEmbedding(model_name)
