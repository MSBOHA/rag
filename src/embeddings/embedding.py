"""
嵌入生成接口（以批量为默认行为）
说明：
- embed 方法既支持单条字符串也支持字符串列表；内部以批量方式编码。
- 保留 batch_embed 作为向后兼容的别名（直接调用 embed）。
- 直接导入依赖（若环境缺包会抛出 ImportError），不使用 try/except 包裹导入。
- 返回值：单条输入返回 List[float]，批量输入返回 List[List[float]]。
- 支持 device 与 batch_size 参数，默认优先使用 GPU（如可用）。
"""

from typing import List, Iterable, Union, Optional
import numpy as np
import torch
from sentence_transformers import SentenceTransformer


class BaseEmbedding:
    def embed(self, texts: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        raise NotImplementedError


class SentenceTransformerEmbedding(BaseEmbedding):
    def __init__(self, model_name: str = "Qwen/Qwen3-Embedding-0.6B", device: Optional[str] = None, precision: Optional[str] = None):
        """
        model_name: sentence-transformers 模型名或 HF id
        device: "cuda"/"cpu"/None(自动选择)
        precision: 可选 "fp16"/"bf16"/None；仅在 CUDA 下生效，用于降低显存占用
        """
        # 选择设备字符串，SentenceTransformer 接受 "cuda" 或 "cpu"
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = "cuda" if str(device).lower().startswith("cuda") else "cpu"
        # 直接传入 device，让 model 参数放到对应设备
        self.model = SentenceTransformer(model_name, device=self.device)
        # 可选精度：在 GPU 上使用半精度以减少显存并可能提速
        self.precision = (precision or "").lower() if isinstance(precision, str) else None
        if self.device == "cuda" and self.precision in ("fp16", "float16"):
            try:
                self.model = self.model.half()
            except Exception:
                # 某些模块不支持 half，忽略
                pass
        elif self.device == "cuda" and self.precision in ("bf16", "bfloat16"):
            try:
                # bfloat16 兼容性取决于 GPU 架构
                for p in self.model.parameters(recurse=True):
                    p.data = p.data.to(torch.bfloat16)
            except Exception:
                pass

    def embed(
    self,
    texts: Union[str, List[str]],
    batch_size: int = 32,
    normalize: bool = True,
    convert_to_tensor: bool = True,
    return_numpy: bool = True,
    force_empty_cache: bool = False,  # 可选：是否在释放后强制调用 empty_cache()
) -> np.ndarray:
        single_input = False
        if isinstance(texts, str):
            texts = [texts]
            single_input = True

        # 在模型上禁用梯度（避免额外显存）
        try:
            with torch.no_grad():
                embs = self.model.encode(
                    texts,
                    batch_size=batch_size,
                    convert_to_tensor=convert_to_tensor,
                    normalize_embeddings=normalize,
                    show_progress_bar=False,
                )
        except RuntimeError as e:
            # 如果 GPU OOM，可以回退到 CPU 模式（较慢但更稳健）
            if 'out of memory' in str(e).lower():
                # 尝试释放一些缓存后再用 CPU 路径
                try:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass
                embs = self.model.encode(
                    texts,
                    batch_size=max(1, batch_size // 2),
                    convert_to_tensor=False,
                    normalize_embeddings=normalize,
                    show_progress_bar=False,
                )
            else:
                raise

        # 将返回统一为 CPU 上的 numpy.float32 ndarray
        if isinstance(embs, torch.Tensor):
            try:
                # 输出统一转为 float32（FAISS 需要），即使内部计算使用半精度
                arr = embs.detach().cpu().to(torch.float32).numpy()
            finally:
                # 删除引用并尽量回收
                try:
                    del embs
                except Exception:
                    pass
                try:
                    import gc
                    gc.collect()
                except Exception:
                    pass
                # 可选：在内存紧张时额外同步并回收 CUDA cache
                if force_empty_cache and torch.cuda.is_available():
                    try:
                        torch.cuda.synchronize()
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
        else:
            # 可能已经是 list 或 numpy
            arr = np.asarray(embs, dtype=np.float32)

        # 确保二维
        if arr.ndim == 1:
            arr = np.expand_dims(arr, 0)

        # 按你的需求返回 numpy ndarray（批量/单条皆为二维）
        return arr
    def batch_embed(
        self,
        texts: Iterable[str],
        batch_size: int = 32,
        normalize: bool = True,
        convert_to_tensor: bool = False,
    ) -> List[List[float]]:
        """
        向后兼容方法：显式批量编码接口（等同于调用 embed 返回批量结果）
        """
        return self.embed(list(texts), batch_size=batch_size, normalize=normalize, convert_to_tensor=convert_to_tensor)


def get_embedder(model_name: str = "Qwen/Qwen3-Embedding-0.6B", device: Optional[str] = None, precision: Optional[str] = None) -> SentenceTransformerEmbedding:
    """
    工厂函数，返回 SentenceTransformerEmbedding 实例
    """
    return SentenceTransformerEmbedding(model_name=model_name, device=device, precision=precision)