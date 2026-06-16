"""
LangGraph 版本：状态图实现带分支+循环的 RAG Agent

流程（图结构）：
    START
      ↓
  [classify]  判断问题是否清晰
      ↓ 条件边
   ┌──┴──────────┐
[reject]      [retrieve]    检索文档
              ↓
           [generate]       生成答案
              ↓ 条件边
         ┌────┴────┐
      [END]    [generate]   质量差则重试（最多1次）
               ↓
             [END]

LangGraph 的核心：
  - 每个步骤是一个"节点"（普通 Python 函数）
  - 节点之间用"边"连接，边可以是条件的
  - 整个流程的数据放在"State"里流转
  - 天然支持分支和循环
"""

from typing import TypedDict, Literal
from langgraph.graph import StateGraph, END, START
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from config import get_llm, retrieve

llm = get_llm()

# ── 1. State：贯穿整个图的数据结构 ───────────────────────────────
class AgentState(TypedDict):
    question: str
    passages: list[str]
    answer: str
    quality: str        # "good" | "bad"
    retry_count: int
    rejected: bool

# ── 2. 节点函数（每个节点接收 state，返回更新的字段）────────────

def classify_node(state: AgentState) -> AgentState:
    """判断问题是否清晰可回答"""
    print(f"\n[Node: classify] Checking if question is clear...")
    prompt = ChatPromptTemplate.from_messages([
        ("human", "Is this a clear, answerable question? Reply only 'yes' or 'no'.\nQuestion: {question}")
    ])
    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({"question": state["question"]}).strip().lower()
    rejected = "no" in result
    print(f"  → {'REJECTED' if rejected else 'OK'}")
    return {"rejected": rejected}

def reject_node(state: AgentState) -> AgentState:
    """拒绝不清晰的问题"""
    print(f"\n[Node: reject] Question is unclear, refusing to answer.")
    return {"answer": "Your question is unclear. Please rephrase it."}

def retrieve_node(state: AgentState) -> AgentState:
    """检索相关文档"""
    print(f"\n[Node: retrieve] Searching knowledge base...")
    passages = retrieve(state["question"])
    print(f"  → Found {len(passages)} passages")
    return {"passages": passages}

def generate_node(state: AgentState) -> AgentState:
    """生成答案"""
    retry = state.get("retry_count", 0)
    print(f"\n[Node: generate] Generating answer (attempt {retry + 1})...")
    context = "\n".join(f"- {p}" for p in state["passages"])
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer based on context. Be specific and detailed."),
        ("human", "Context:\n{context}\n\nQuestion: {question}")
    ])
    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "question": state["question"]})
    print(f"  → Answer: {answer[:80]}...")

    # 简单质量评估：答案太短则认为质量差
    quality = "bad" if len(answer.split()) < 10 else "good"
    print(f"  → Quality: {quality}")
    return {"answer": answer, "quality": quality, "retry_count": retry + 1}

# ── 3. 条件边函数（返回下一个节点的名字）─────────────────────────

def route_after_classify(state: AgentState) -> Literal["reject", "retrieve"]:
    return "reject" if state["rejected"] else "retrieve"

def route_after_generate(state: AgentState) -> Literal["generate", "__end__"]:
    """答案质量差且还没重试过，则重试"""
    if state["quality"] == "bad" and state.get("retry_count", 0) < 2:
        print(f"\n[Edge] Quality bad, retrying...")
        return "generate"
    return "__end__"

# ── 4. 构建图 ──────────────────────────────────────────────────────
def build_graph():
    graph = StateGraph(AgentState)

    # 添加节点
    graph.add_node("classify", classify_node)
    graph.add_node("reject", reject_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)

    # 添加边
    graph.add_edge(START, "classify")
    graph.add_conditional_edges("classify", route_after_classify)  # 条件边
    graph.add_edge("reject", END)
    graph.add_edge("retrieve", "generate")
    graph.add_conditional_edges("generate", route_after_generate)  # 条件边（可循环）

    return graph.compile()

# ── 5. 运行 ───────────────────────────────────────────────────────
def run(question: str):
    print(f"\n{'='*50}")
    print(f"[LangGraph] Question: {question}")
    print(f"{'='*50}")

    app = build_graph()
    final_state = app.invoke({
        "question": question,
        "passages": [],
        "answer": "",
        "quality": "",
        "retry_count": 0,
        "rejected": False,
    })
    print(f"\n[Final Answer] {final_state['answer']}")
    return final_state["answer"]


if __name__ == "__main__":
    run("What is LangGraph and how is it different from LangChain?")
    print("\n" + "="*50)
    run("????? gibberish question @#$%")   # 应该被 classify 节点拒绝
