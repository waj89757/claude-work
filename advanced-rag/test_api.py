"""测试 API Key 是否可用"""
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
base_url = os.getenv("OPENAI_BASE_URL")
model = os.getenv("LLM_MODEL", "Qwen/Qwen3-Next-80B-A3B-Instruct-FP8")

print(f"API Key: {api_key[:10]}..." if api_key else "API Key: 未设置")
print(f"Base URL: {base_url}")
print(f"Model: {model}")

try:
    client = OpenAI(api_key=api_key, base_url=base_url)
    
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "说一句话测试"}],
        max_tokens=50
    )
    
    print(f"\n✅ API Key 有效!")
    print(f"响应: {response.choices[0].message.content}")
    
except Exception as e:
    print(f"\n❌ API Key 测试失败:")
    print(f"错误: {e}")
