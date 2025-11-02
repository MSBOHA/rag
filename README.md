# 运行方法

## 安装依赖




RAG 多模态检索系统


### 1. 环境准备

- 使用 [uv](https://github.com/astral-sh/uv) 进行依赖管理和虚拟环境：
	```sh
	pip install uv
	uv sync
	```
- 项目依赖已在 `pyproject.toml` 中声明，无需手动 pip 安装。

### 2. .env 配置

- 在项目根目录下创建 `.env` 文件，内容示例：
	```ini
	DASHSCOPE_API_KEY=你的阿里云API Key
    GEMINI_API_KEY=你的GEMINI API Key

	# 其他API Key可按需添加
	```
api申请地址：
https://www.aliyun.com/product/bailian
https://aistudio.google.com/apikey

### 3. 多模态文档支持

- 支持文本、PDF、图片等多模态文档。
- PDF 解析依赖 `pymupdf`，图片 OCR 依赖 `pytesseract`。
- 需自行安装 [Google Tesseract OCR](https://github.com/tesseract-ocr/tesseract) 并配置好环境变量，确保命令行可用 `tesseract`。
- 中文识别需下载 `chi_sim.traineddata` 并放入 tesseract 的 tessdata 目录。

### 4. 建库与查询

- 运行 `python -m scripts.build_index`，选择数据文件夹批量建库。
- 运行 `python -m scripts.query`，选择库并输入问题进行检索。

