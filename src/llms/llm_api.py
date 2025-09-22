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

    def generate(self, query: str, chunks: List[str]) -> str:
        prompt = f"""你是一位知识助手，请根据用户的问题和下列片段生成准确的回答。\n\n用户问题: {query}\n\n相关片段:\n{"\n\n".join(chunks)}\n\n请基于上述内容作答，不要编造信息。"""
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

    def generate(self, query: str, chunks: List[str]) -> str:
        prompt = f"你是一位知识助手，请根据用户的问题和下列片段生成准确的回答。\n\n用户问题: {query}\n\n相关片段:\n{'\n\n'.join(chunks)}\n\n请基于上述内容作答，不要编造信息。"
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            stream=False
        )
        # 兼容openai返回格式
        return completion.choices[0].message.content

def get_llm(model_name: str = "gemini-2.5-flash") -> BaseLLM:
    if model_name.startswith("qwen"):
        return QwenLLM(model_name)
    return GeminiLLM(model_name)
