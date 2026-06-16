"""
LangGraph 多 Agent 示例：Supervisor 模式

场景：用户提一个需求，Supervisor 分配给专门的 Agent 处理
  - ResearchAgent：负责查资料（检索知识库）
  - WriterAgent：负责写内容（把资料整理成文章）
  - ReviewAgent：负责审核（检查内容质量，决定是否通过）

图结构：
    START
      ↓
  [supervisor]  分析任务，决定派给谁
      ↓ 条件边
   ┌──┼──────────────┐
[research]  [writer]  [review]
      ↓         ↓         ↓
  [supervisor]（汇报结果，Supervisor 再决定下一步）
      ↓
   ┌──┴──┐
 [END]  继续派任务
"""

from typing import TypedDict, Literal, Annotated
import operator
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END, START
from config import get_llm, retrieve

llm = get_llm()

# ── 1. State ──────────────────────────────────────────────────────
class MultiAgentState(TypedDict):
    task: str                                       # 用户原始需求
    research_result: str                            # ResearchAgent 产出
    draft: str                                      # WriterAgent 产出
    review_feedback: str                            # ReviewAgent 产出
    next: str                                       # Supervisor 决定下一步去哪
    messages: Annotated[list[str], operator.add]    # 各 Agent 的日志（追加模式）

# ── 2. Supervisor Agent ───────────────────────────────────────────
def supervisor_node(state: MultiAgentState) -> MultiAgentState:
    """
    Supervisor：分析当前状态，决定下一步交给谁。
    这是多 Agent 的核心：由一个 LLM 做路由决策。
    """
    print(f"\n{'─'*40}")
    print(f"[Supervisor] Analyzing situation...")

    # 构建 Supervisor 的上下文
    status = f"""
Task: {state['task']}
Research done: {'Yes' if state['research_result'] else 'No'}
Draft written: {'Yes' if state['draft'] else 'No'}
Review done: {'Yes' if state['review_feedback'] else 'No'}
"""

    # 明确判断 review 结果，避免 Supervisor LLM 误判
    review_feedback = state.get("review_feedback", "")
    review_approved = review_feedback.upper().startswith("APPROVED")
    review_rejected = review_feedback.upper().startswith("REJECTED")

    # 直接用规则路由，不依赖 LLM（LLM 路由在复杂状态下容易出错）
    if not state["research_result"]:
        decision = "research"
    elif not state["draft"]:
        decision = "writer"
    elif not review_feedback:
        decision = "review"
    elif review_approved:
        decision = "FINISH"
    elif review_rejected:
        decision = "writer"
    else:
        decision = "FINISH"

    # 仅做日志打印用的 prompt，不影响路由
    prompt = None

    print(f"  → Decision: {decision}")
    return {
        "next": decision,
        "messages": [f"[Supervisor] Next step: {decision}"]
    }

# ── 3. Research Agent ─────────────────────────────────────────────
def research_node(state: MultiAgentState) -> MultiAgentState:
    """检索知识库，整理成研究报告"""
    print(f"\n[ResearchAgent] Researching: {state['task'][:50]}...")

    passages = retrieve(state["task"], k=3)
    context = "\n".join(f"• {p}" for p in passages)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a researcher. Summarize the key facts from these passages."),
        ("human", f"Task: {state['task']}\n\nPassages:\n{context}")
    ])
    result = (prompt | llm | StrOutputParser()).invoke({})
    print(f"  → Research complete ({len(result.split())} words)")

    return {
        "research_result": result,
        "messages": [f"[ResearchAgent] Research complete"]
    }

# ── 4. Writer Agent ───────────────────────────────────────────────
def writer_node(state: MultiAgentState) -> MultiAgentState:
    """基于研究结果写一篇文章"""
    print(f"\n[WriterAgent] Writing draft...")

    feedback_hint = ""
    if state.get("review_feedback"):
        feedback_hint = f"\n\nPrevious review feedback (please address these issues):\n{state['review_feedback']}"

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a technical writer. Write a clear, concise article."),
        ("human", f"Task: {state['task']}\n\nResearch:\n{state['research_result']}{feedback_hint}")
    ])
    draft = (prompt | llm | StrOutputParser()).invoke({})
    print(f"  → Draft written ({len(draft.split())} words)")

    return {
        "draft": draft,
        "review_feedback": "",   # 重置，让 Supervisor 知道需要重新 review
        "messages": [f"[WriterAgent] Draft written"]
    }

# ── 5. Review Agent ───────────────────────────────────────────────
def review_node(state: MultiAgentState) -> MultiAgentState:
    """审核文章质量，给出 APPROVED 或 REJECTED"""
    print(f"\n[ReviewAgent] Reviewing draft...")

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a quality reviewer. Review the draft strictly.
If the draft is clear, accurate, and well-structured: start with 'APPROVED:'
If it needs improvement: start with 'REJECTED:' and explain why."""),
        ("human", f"Task: {state['task']}\n\nDraft:\n{state['draft']}")
    ])
    feedback = (prompt | llm | StrOutputParser()).invoke({})
    approved = feedback.upper().startswith("APPROVED")
    print(f"  → {'✅ APPROVED' if approved else '❌ REJECTED'}")
    if not approved:
        print(f"  → Feedback: {feedback[:100]}...")

    return {
        "review_feedback": feedback,
        "messages": [f"[ReviewAgent] {'APPROVED' if approved else 'REJECTED'}"]
    }

# ── 6. 条件边：Supervisor 决定去哪个节点 ──────────────────────────
def route_by_supervisor(state: MultiAgentState) -> Literal["research", "writer", "review", "__end__"]:
    next_step = state.get("next", "research")
    if next_step == "FINISH":
        return "__end__"
    return next_step

# ── 7. 构建图 ──────────────────────────────────────────────────────
graph = StateGraph(MultiAgentState)

graph.add_node("supervisor", supervisor_node)
graph.add_node("research", research_node)
graph.add_node("writer", writer_node)
graph.add_node("review", review_node)

# 所有 Agent 完成后都回到 Supervisor 汇报
graph.add_edge(START, "supervisor")
graph.add_conditional_edges("supervisor", route_by_supervisor)
graph.add_edge("research", "supervisor")   # Research 完 → 汇报 Supervisor
graph.add_edge("writer", "supervisor")     # Writer 完 → 汇报 Supervisor
graph.add_edge("review", "supervisor")     # Review 完 → 汇报 Supervisor

app = graph.compile()

# ── 8. 运行 ───────────────────────────────────────────────────────
if __name__ == "__main__":
    task = "Write a short article explaining what LangGraph is and why it's better than LangChain for complex agents."

    print(f"{'='*50}")
    print(f"Task: {task}")
    print(f"{'='*50}")

    result = app.invoke({
        "task": task,
        "research_result": "",
        "draft": "",
        "review_feedback": "",
        "next": "",
        "messages": [],
    })

    print(f"\n{'='*50}")
    print("【执行日志】")
    print(f"{'='*50}")
    for msg in result["messages"]:
        print(f"  {msg}")

    print(f"\n{'='*50}")
    print("【最终文章】")
    print(f"{'='*50}")
    print(result["draft"])
