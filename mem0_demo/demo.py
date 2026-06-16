"""
Mem0 Demo：带长期记忆的对话助手

Mem0 做的事：
- 自动从对话中抽取重要信息存成"记忆"
- 下次对话时自动检索相关记忆注入 prompt
- 用户不用每次重复自我介绍

本 demo 用本地 SQLite + 本地向量存储（不需要外部服务），
LLM 调用复用 dspy_quickstart 里的配置。

运行：cd mem0_demo && python demo.py
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# 加载 .env（复用 dspy_quickstart 的配置）
load_dotenv(Path(__file__).parent.parent / "dspy_quickstart" / ".env")

from mem0 import Memory

# 修复内网网关不支持 max_tokens 参数的问题
import mem0.llms.openai as _openai_llm
_orig_get_common_params = _openai_llm.OpenAILLM._get_common_params
def _patched_get_common_params(self, **kwargs):
    params = _orig_get_common_params(self, **kwargs)
    params.pop("max_tokens", None)   # 内网网关用 max_completion_tokens，不接受 max_tokens
    params.pop("top_p", None)
    return params
_openai_llm.OpenAILLM._get_common_params = _patched_get_common_params

# ── 1. 配置 Mem0（本地模式，不需要 Mem0 云账号）─────────────────
config = {
    "llm": {
        "provider": "openai",
        "config": {
            "model": os.getenv("LLM_MODEL"),
            "api_key": os.getenv("LLM_API_KEY"),
            "openai_base_url": os.getenv("LLM_API_BASE"),
            "max_tokens": None,  # 内网网关用 max_completion_tokens，不传 max_tokens
        }
    },
    "embedder": {
        "provider": "huggingface",
        "config": {
            "model": "sentence-transformers/all-MiniLM-L6-v2",  # 本地模型，不需要 API key
        }
    },
    "vector_store": {
        "provider": "chroma",        # 本地向量库，无需额外服务
        "config": {
            "collection_name": "mem0_demo",
            "path": "./chroma_db",   # 持久化到本地目录
        }
    },
}

memory = Memory.from_config(config)

USER_ID = "user_waj"   # 每个用户有独立的记忆空间

# ── 2. 模拟"第一次对话"：告诉 AI 一些个人信息 ─────────────────────
print("=" * 55)
print("【第一次对话】告诉 AI 你的信息")
print("=" * 55)

first_conversation = [
    {"role": "user",      "content": "我叫王安节，是一个技术团队负责人，负责海外业务。"},
    {"role": "assistant", "content": "你好王安节！很高兴认识你，有什么我能帮你的吗？"},
    {"role": "user",      "content": "我们团队主要做巴西和印尼市场，我最近在学习 DSPy 和 LangGraph。"},
    {"role": "assistant", "content": "明白了，你在做海外市场扩展，同时在提升 AI 技术栈，很不错！"},
]

# add() 会让 LLM 自动从对话中抽取记忆
result = memory.add(first_conversation, user_id=USER_ID)

print("\n✅ 第一次对话完成，Mem0 自动抽取的记忆：")
for item in result.get("results", []):
    print(f"  [{item['event']}] {item['memory']}")

# ── 3. 查看当前存储的所有记忆 ──────────────────────────────────────
print("\n" + "=" * 55)
print("【当前记忆库】")
print("=" * 55)

all_memories = memory.get_all(filters={"user_id": USER_ID})
for i, mem in enumerate(all_memories.get("results", []), 1):
    print(f"  {i}. {mem['memory']}")

# ── 4. 模拟"第二次对话"：不提任何背景，看 AI 是否记得 ─────────────
print("\n" + "=" * 55)
print("【第二次对话】换了个新会话，不提任何背景")
print("=" * 55)

new_question = "我想学习下 RAG，你有什么建议？"
print(f"\n用户：{new_question}")

# 根据问题检索相关记忆
relevant_memories = memory.search(query=new_question, filters={"user_id": USER_ID})

print("\n🔍 Mem0 检索到的相关记忆：")
for mem in relevant_memories.get("results", []):
    print(f"  - {mem['memory']}  (相关度: {mem.get('score', 'N/A'):.3f})")

# 把记忆注入 prompt
memory_context = "\n".join(
    f"- {m['memory']}" for m in relevant_memories.get("results", [])
)

system_prompt = f"""你是一个 AI 助手。
以下是你已知的关于这个用户的信息：
{memory_context}

请结合用户背景给出个性化建议。"""

print(f"\n📋 注入到 prompt 的记忆上下文：")
print(memory_context)

# 实际调用 LLM（用 openai 库直接调）
from openai import OpenAI
client = OpenAI(
    api_key=os.getenv("LLM_API_KEY"),
    base_url=os.getenv("LLM_API_BASE"),
)

response = client.chat.completions.create(
    model=os.getenv("LLM_MODEL"),
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": new_question},
    ]
)
answer = response.choices[0].message.content

print(f"\n🤖 AI 回答（带记忆个性化）：\n{answer}")

# ── 5. 第二次对话也存入记忆 ───────────────────────────────────────
memory.add(
    [
        {"role": "user",      "content": new_question},
        {"role": "assistant", "content": answer},
    ],
    user_id=USER_ID
)

print("\n" + "=" * 55)
print("【最终记忆库】（两次对话后）")
print("=" * 55)
all_memories = memory.get_all(filters={"user_id": USER_ID})
for i, mem in enumerate(all_memories.get("results", []), 1):
    print(f"  {i}. {mem['memory']}")

print("\n✅ Demo 完成！记忆已持久化到 ./chroma_db，下次运行仍然有效。")
print("   如需清空记忆，删除 ./chroma_db 目录即可。")
