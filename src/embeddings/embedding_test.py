import argparse
import time
import os
import subprocess
import sys

import torch

# 获取 nvidia-smi 的显存信息（返回每个 GPU 的 "used,total" 行）
def nvidia_smi_mem():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        lines = [line.strip() for line in out.splitlines() if line.strip()]
        return lines
    except Exception:
        return None

# 打印当前选定设备和显存信息（优先使用 torch 的接口，再调用 nvidia-smi）
def print_gpu_info(device):
    print("Torch CUDA available:", torch.cuda.is_available())
    # 判断设备是否为 cuda
    if device.type == torch.device("cuda").type:
        print("使用设备:", device)
        print("torch.cuda.device_count():", torch.cuda.device_count())
        try:
            idx = device.index if device.index is not None else 0
            print("当前 CUDA 设备索引:", idx, torch.cuda.get_device_name(idx))
        except Exception:
            pass
        # 打印当前已分配和历史峰值显存（可能为 0，如果尚未分配）
        try:
            print("torch.cuda.memory_allocated():", torch.cuda.memory_allocated(device))
            print("torch.cuda.max_memory_allocated():", torch.cuda.max_memory_allocated(device))
        except Exception:
            pass
        smi = nvidia_smi_mem()
        if smi:
            print("nvidia-smi 显存 (used,total) 每 GPU:")
            for line in smi:
                print("  ", line)
    else:
        print("使用 CPU 设备")

# 加载 embedding 模型：优先使用 sentence-transformers，失败后回退到 transformers HF AutoModel
def load_model(model_name, device):
    # 优先尝试 sentence-transformers（更方便的 encode 接口）
    try:
        from sentence_transformers import SentenceTransformer
        print("正在加载 SentenceTransformer:", model_name)
        # 允许在构造时传入 device 字符串，例如 "cuda" 或 "cpu"
        model = SentenceTransformer(model_name, device=str(device))
        return model, "sentence-transformers"
    except Exception as e:
        print("sentence-transformers 不可用或加载失败:", e)
    # 回退到 HuggingFace AutoModel（需要自行实现 pooling）
    try:
        from transformers import AutoTokenizer, AutoModel
        print("正在加载 HuggingFace 模型:", model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name).to(device)
        return (tokenizer, model), "hf"
    except Exception as e:
        print("加载 HuggingFace 模型失败:", e)
    raise RuntimeError("未检测到可用的 embedding 后端。请安装 sentence-transformers 或 transformers。")

# 对一批文本做编码，返回 torch.Tensor，确保在 target device 上
def encode_batch(model_info, backend, texts, device):
    if backend == "sentence-transformers":
        model = model_info
        # 使用 sentence-transformers 的 convert_to_tensor=True，直接得到 torch.Tensor（通常在 CPU 或 GPU）
        # show_progress_bar=False 避免进度条干扰批量测量
        embs = model.encode(texts, batch_size=len(texts), convert_to_tensor=True, show_progress_bar=False)
        # embs 可能已经是 torch.Tensor；若不是则进行安全转换并移动到目标 device
        if isinstance(embs, torch.Tensor):
            try:
                return embs.to(device)
            except Exception:
                return embs.cpu().to(device)
        # 若返回 numpy 数组或其它类型，尝试转换
        try:
            import numpy as np
            arr = np.asarray(embs)
            return torch.from_numpy(arr).to(device)
        except Exception:
            # 兜底处理：逐个转换并堆叠
            try:
                tensors = [torch.as_tensor(x) for x in embs]
                return torch.stack(tensors).to(device)
            except Exception as e:
                raise RuntimeError("无法将 sentence-transformers 输出转换为 torch.Tensor: " + str(e))
    else:
        # HuggingFace AutoModel 分支：做 padding + mean pooling（在 token 非 pad 的位置求均值）
        tokenizer, model = model_info
        enc = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**enc, return_dict=True)
            # mask 用于忽略 pad token（注意 tokenizer.pad_token_id 可能为 None）
            pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
            toks = enc["input_ids"].ne(pad_id).unsqueeze(-1)
            embeddings = (out.last_hidden_state * toks).sum(1) / toks.sum(1)
            return embeddings

# 主流程：命令行参数、设备选择、样本加载、模型加载、warmup、测量吞吐与显存
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-Embedding-0.6B", help="模型名（sentence-transformers 或 HF）")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_samples", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--device", default="auto", choices=["auto","cpu","cuda"])
    parser.add_argument("--text_file", default="data/eval/eval.txt", help="可选：作为样本的文本文件")
    args = parser.parse_args()

    # 选择设备：cpu / cuda / auto
    if args.device == "cpu":
        device = torch.device("cpu")
    elif args.device == "cuda":
        device = torch.device("cuda")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("选定设备:", device)
    print_gpu_info(device)

    # 加载文本样本：优先从指定文件按段落读取，若数量不足则循环重复以达到 num_samples
    texts = []
    if os.path.exists(args.text_file):
        with open(args.text_file, "r", encoding="utf-8") as f:
            raw = f.read().strip()
            # 优先按双换行分段为段落
            paras = [p.strip() for p in raw.split("\n\n") if p.strip()]
            if len(paras) == 0:
                paras = [l.strip() for l in raw.splitlines() if l.strip()]
            if len(paras) == 0:
                paras = [raw]
            # 重复段落直到满足样本数
            while len(texts) < args.num_samples:
                for p in paras:
                    texts.append(p)
                    if len(texts) >= args.num_samples:
                        break
    else:
        # 若文件不存在，生成简易合成样本
        print("文本文件不存在，生成合成文本样本")
        base = ("人工智能是计算机科学的一个分支，旨在开发模拟和扩展人类智能的系统。"
                "它包括机器学习、深度学习、自然语言处理等领域。")
        for i in range(args.num_samples):
            texts.append(f"{base} 示例编号：{i}")

    texts = texts[: args.num_samples]
    print(f"准备 {len(texts)} 条样本，batch_size={args.batch_size}")

    # 加载模型（sentence-transformers 优先）
    model_info, backend = load_model(args.model, device)
    print("后端:", backend)

    # 快速检查模型参数所在设备（仅在 HF 分支或 model 对象暴露 device 时有意义）
    try:
        if backend == "hf":
            _, model = model_info
            first_param = next(model.parameters())
            print("模型首参数所在设备:", first_param.device)
        else:
            # sentence-transformers Module 可能有 .device 属性
            if hasattr(model_info, "device"):
                print("模型报告的 device:", model_info.device)
    except Exception:
        pass

    # warmup 请求若干批次，确保模型被加载并触发 JIT/cuda 分配
    print("warmup ...")
    for i in range(min(args.warmup, len(texts)//args.batch_size or 1)):
        batch = texts[i*args.batch_size:(i+1)*args.batch_size]
        _ = encode_batch(model_info, backend, batch, device)
        if device.type == "cuda":
            torch.cuda.synchronize()

    # 重置 PyTorch 的峰值显存统计（如果可用）
    if device.type == "cuda":
        try:
            torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass

    # 逐批测量编码时间并输出每批吞吐
    total = 0
    start = time.perf_counter()
    for i in range(0, len(texts), args.batch_size):
        batch = texts[i:i+args.batch_size]
        t0 = time.perf_counter()
        emb = encode_batch(model_info, backend, batch, device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        elapsed = t1 - t0
        total += len(batch)
        # 输出单批时间和吞吐（samples/s）
        print(f"batch {i//args.batch_size+1}: size={len(batch)} time={elapsed:.4f}s  {len(batch)/elapsed:.2f} samples/s")
    end = time.perf_counter()
    total_time = end - start
    # 输出总耗时、平均吞吐和平均单样本延迟
    print(f"总耗时: {total_time:.4f}s, 总样本: {total}, 平均吞吐: {total/total_time:.2f} samples/s, 平均单样本延迟: {total_time/total:.4f}s")

    # 输出运行后显存信息以便参考
    print("=== 运行后显存信息 ===")
    print_gpu_info(device)
    print("done.")

if __name__ == "__main__":
    main()