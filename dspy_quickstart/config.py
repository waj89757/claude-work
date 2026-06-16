"""
LLM 配置加载模块
从 .env 文件读取配置，统一管理 LM 实例
"""

import os
from pathlib import Path
from dotenv import load_dotenv
import dspy

# 加载 .env 文件
env_path = Path(__file__).parent / ".env"
load_dotenv(env_path)

def get_lm():
    """获取配置好的 LM 实例"""
    api_key = os.getenv("LLM_API_KEY")
    api_base = os.getenv("LLM_API_BASE")
    model = os.getenv("LLM_MODEL", "gpt-4o-mini")
    
    if not api_key:
        raise ValueError("请在 .env 文件中配置 LLM_API_KEY")
    if not api_base:
        raise ValueError("请在 .env 文件中配置 LLM_API_BASE")
    
    # DSPy 使用 openai/ 前缀表示兼容 OpenAI 格式的 API
    lm = dspy.LM(
        f"openai/{model}",
        api_base=api_base,
        api_key=api_key,
    )
    return lm

def configure_dspy():
    """配置 DSPy 全局 LM"""
    lm = get_lm()
    dspy.configure(lm=lm)
    return lm
