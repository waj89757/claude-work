"""公共配置：LLM + 知识库"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

def get_llm():
    return ChatOpenAI(
        model=os.getenv("LLM_MODEL"),
        api_key=os.getenv("LLM_API_KEY"),
        base_url=os.getenv("LLM_API_BASE"),
        temperature=0,
    )

# 模拟知识库（替代向量数据库）
KNOWLEDGE_BASE = [
    "LangChain is a framework for building LLM applications using composable chains.",
    "LangGraph is built on top of LangChain and models agent workflows as a state graph with nodes and edges.",
    "LangGraph supports cycles and branching, making it ideal for multi-step agent loops.",
    "LangChain's LCEL (LangChain Expression Language) uses the pipe operator | to chain components.",
    "A LangGraph node is a Python function that takes state and returns updated state.",
    "LangGraph edges can be conditional, routing to different nodes based on state values.",
    "RAG (Retrieval-Augmented Generation) retrieves relevant documents before generating an answer.",
    "DSPy is a framework that optimizes LLM prompts automatically using training data.",
]

def retrieve(query: str, k: int = 3) -> list[str]:
    """关键词匹配检索（模拟向量检索）"""
    query_words = set(query.lower().split())
    scored = [(len(query_words & set(doc.lower().split())), doc) for doc in KNOWLEDGE_BASE]
    scored.sort(reverse=True)
    return [doc for _, doc in scored[:k]]
