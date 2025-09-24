"""
LLM接口和实现
"""


from typing import List
from dotenv import load_dotenv
import os
from openai import OpenAI
from google import genai

class BaseLLM:
    def generate(self, query: str, chunks: List[str]) -> str:
        raise NotImplementedError


# Gemini LLM
class GeminiLLM(BaseLLM):
    def __init__(self, model_name: str = "gemini-2.5-flash"):
        load_dotenv()
        self.client = genai.Client()
        self.model_name = model_name

    def generate(self, query: str, chunks: List[str], messages: list = None) -> str:
        # 可选：拼接历史messages到prompt
        history = ""
        if messages:
            for m in messages:
                role = m.get("role", "user")
                content = m.get("content", "")
                history += f"{role}: {content}\n"
        prompt = f"{history}\n用户问题: {query}\n\n相关片段:\n{'\n\n'.join(chunks)}\n\n请基于上述内容作答，不要编造信息。"
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt
        )
        return response.text

# Qwen LLM (Aliyun DashScope)
class QwenLLM(BaseLLM):
    def __init__(self, model_name: str = "qwen-plus"):
        load_dotenv()
        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            raise ValueError("请在.env或环境变量中配置DASHSCOPE_API_KEY")
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        self.model_name = model_name

    def generate(self, query: str, chunks: List[str], messages: list = None) -> str:
        # 如果传入messages则直接用，否则兼容原RAG模式
        if messages:
            stream = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                stream=True
            )
            content = ""
            for chunk in stream:
                delta = getattr(chunk.choices[0].delta, "content", None)
                if delta:
                    content += delta
            return content
        else:
            system_prompt = (
                "你是一位知识助手，请严格根据下列检索到的文档片段回答用户问题，不要编造信息，也不要泛化。"
                "不要使用外部知识。如果没提到，就说‘未提及’。回答时请引用片段编号或内容。"
                "如无相关内容请直接说明。"
            )
            user_prompt = (
                f"用户问题: {query}\n\n相关片段（编号从0开始）：\n"
                + '\n\n'.join([f"[{i}] {chunk}" for i, chunk in enumerate(chunks)])
                + "\n\n请基于上述片段作答，如无相关内容请说明。"
            )
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            stream = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                stream=True
            )
            content = ""
            for chunk in stream:
                delta = getattr(chunk.choices[0].delta, "content", None)
                if delta:
                    content += delta
            return content

def get_llm(model_name: str = "gemini-2.5-flash") -> BaseLLM:
    if model_name.startswith("qwen"):
        return QwenLLM(model_name)
    return GeminiLLM(model_name)
