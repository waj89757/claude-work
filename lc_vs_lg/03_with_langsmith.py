"""
LangSmith 集成示例：在 LangGraph Agent 上加追踪

LangSmith 的接入方式极简：只需设置 3 个环境变量，
LangChain/LangGraph 的每次 LLM 调用、节点执行都会自动上报。
不需要修改任何业务代码。

需要先注册：https://smith.langchain.com （免费）
然后在项目设置里获取 API Key，填入 .env 文件
"""

import os
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

# ── LangSmith 配置（就这 3 行，其余代码完全不变）─────────────────
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY", "your-langsmith-key")
os.environ["LANGCHAIN_PROJECT"] = "lc-vs-lg-demo"   # 在 LangSmith 上显示的项目名

# ── 以下代码和 02_langgraph_agent.py 完全一样 ─────────────────────
from typing import TypedDict, Literal
from langgraph.graph import StateGraph, END, START
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from config import get_llm, retrieve

llm = get_llm()

class AgentState(TypedDict):
    question: str
    passages: list[str]
    answer: str
    quality: str
    retry_count: int
    rejected: bool

def classify_node(state: AgentState) -> AgentState:
    prompt = ChatPromptTemplate.from_messages([
        ("human", "Is this a clear, answerable question? Reply only 'yes' or 'no'.\nQuestion: {question}")
    ])
    result = (prompt | llm | StrOutputParser()).invoke({"question": state["question"]}).strip().lower()
    return {"rejected": "no" in result}

def reject_node(state: AgentState) -> AgentState:
    return {"answer": "Your question is unclear. Please rephrase it."}

def retrieve_node(state: AgentState) -> AgentState:
    return {"passages": retrieve(state["question"])}

def generate_node(state: AgentState) -> AgentState:
    context = "\n".join(f"- {p}" for p in state["passages"])
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer based on context."),
        ("human", "Context:\n{context}\n\nQuestion: {question}")
    ])
    answer = (prompt | llm | StrOutputParser()).invoke({"context": context, "question": state["question"]})
    quality = "bad" if len(answer.split()) < 10 else "good"
    return {"answer": answer, "quality": quality, "retry_count": state.get("retry_count", 0) + 1}

def route_after_classify(state: AgentState) -> Literal["reject", "retrieve"]:
    return "reject" if state["rejected"] else "retrieve"

def route_after_generate(state: AgentState) -> Literal["generate", "__end__"]:
    if state["quality"] == "bad" and state.get("retry_count", 0) < 2:
        return "generate"
    return "__end__"

graph = StateGraph(AgentState)
graph.add_node("classify", classify_node)
graph.add_node("reject", reject_node)
graph.add_node("retrieve", retrieve_node)
graph.add_node("generate", generate_node)
graph.add_edge(START, "classify")
graph.add_conditional_edges("classify", route_after_classify)
graph.add_edge("reject", END)
graph.add_edge("retrieve", "generate")
graph.add_conditional_edges("generate", route_after_generate)
app = graph.compile()

# ── 运行（LangSmith 自动追踪每次调用）────────────────────────────
if __name__ == "__main__":
    result = app.invoke({
        "question": "What is LangGraph and how is it different from LangChain?",
        "passages": [], "answer": "", "quality": "", "retry_count": 0, "rejected": False,
    })
    print(f"\n[Answer] {result['answer']}")
    print(f"\n✅ 去 https://smith.langchain.com 查看完整追踪链路")
    print(f"   项目名：lc-vs-lg-demo")
    print(f"   可以看到每个节点的输入输出、LLM 调用、耗时、token 消耗")
