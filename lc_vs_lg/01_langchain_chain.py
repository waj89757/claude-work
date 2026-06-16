"""
LangChain 版本：线性 Chain 实现 RAG

流程：问题 → 检索 → 生成答案

问题：如果要加"判断问题是否清晰"、"答案质量差则重试"等逻辑，
      就必须在 Python 层面手动 if/else，Chain 本身无法表达分支和循环。
      这就是 LangChain 的局限，也是 LangGraph 出现的原因。
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from config import get_llm, retrieve

llm = get_llm()

# ── Prompt 模板 ────────────────────────────────────────────────────
rag_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Answer based on the context provided."),
    ("human", """Context:
{context}

Question: {question}

Answer concisely based on the context.""")
])

# ── Chain：用 | 管道符把组件串起来 ────────────────────────────────
#    input dict → prompt → llm → str
rag_chain = rag_prompt | llm | StrOutputParser()

def run(question: str):
    print(f"\n{'='*50}")
    print(f"[LangChain] Question: {question}")
    print(f"{'='*50}")

    # Step 1: 检索
    passages = retrieve(question)
    context = "\n".join(f"- {p}" for p in passages)
    print(f"\n[Retrieve] Got {len(passages)} passages")

    # Step 2: 生成（Chain 调用）
    answer = rag_chain.invoke({"context": context, "question": question})
    print(f"\n[Answer] {answer}")

    # ⚠️ 如果这里要加"答案质量差则重试"逻辑，只能在 Python 里手写 if/for
    # 这不是 Chain 的能力，是你自己的控制流——Chain 对此无能为力
    return answer


if __name__ == "__main__":
    run("What is LangGraph and how is it different from LangChain?")
    run("What is DSPy?")
