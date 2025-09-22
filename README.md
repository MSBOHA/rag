# 运行方法

## 安装依赖

首先请确保你的系统已经安装了 uv 和 Jupyter，否则请参照如下链接安装：


RAG 多模态检索系统

## 快速开始

### 1. 环境准备

- 推荐使用 [uv](https://github.com/astral-sh/uv) 进行依赖管理和虚拟环境：
	```sh
	pip install uv
	uv sync
	```
- 项目依赖已在 `pyproject.toml` 中声明，无需手动 pip 安装。

### 2. .env 配置

- 在项目根目录下创建 `.env` 文件，内容示例：
	```ini
	DASHSCOPE_API_KEY=你的阿里云API Key
    https://www.aliyun.com/product/bailian
    GEMINI_API_KEY=你的GEMINI API Key
    https://aistudio.google.com/apikey
	# 其他API Key可按需添加
	```


### 3. 多模态文档支持

- 支持文本、PDF、图片等多模态文档。
- PDF 解析依赖 `pymupdf`，图片 OCR 依赖 `pytesseract`。
- 需自行安装 [Google Tesseract OCR](https://github.com/tesseract-ocr/tesseract) 并配置好环境变量，确保命令行可用 `tesseract`。
- 中文识别需下载 `chi_sim.traineddata` 并放入 tesseract 的 tessdata 目录。

### 4. 建库与查询

- 运行 `python -m scripts.build_index`，选择数据文件夹批量建库。
- 运行 `python -m scripts.query`，选择库并输入问题进行检索。

### 5. 其他说明

- LLM 支持阿里云 DashScope（Qwen）、Google Gemini 等，需在 `.env` 配置对应 API Key。
- 支持多数据集、多库管理，索引和数据自动按文件夹归档。



本项目基于 MarkTechStation 的 RAG 系统实现（https://github.com/MarkTechStation/VideoCode.git）
