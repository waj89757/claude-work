"""列出可用模型"""
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL")
)

try:
    models = client.models.list()
    print("可用模型列表:")
    for model in models.data[:20]:  # 只显示前20个
        print(f"  - {model.id}")
except Exception as e:
    print(f"获取模型列表失败: {e}")
    print("\n尝试直接测试常见模型名称...")
    
    test_models = ["gpt-4", "gpt-3.5-turbo", "claude-3-sonnet", "qwen-plus", "glm-4"]
    for model in test_models:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=5
            )
            print(f"  ✅ {model} 可用")
        except:
            print(f"  ❌ {model} 不可用")
