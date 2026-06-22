"""
RLM (Recursive Language Model) 递归语言模型 Demo
=================================================

什么是 RLM？
-----------
RLM 是一种让 LLM 递归调用自身来解决复杂问题的模式。
核心思想：当一个问题太复杂时，LLM 把它拆解成子问题，
每个子问题再次调用 LLM 解决，直到问题足够简单可以直接回答。

结构类比：
  传统 LLM 调用：问题 → LLM → 答案（一次）
  RLM：          问题 → LLM → [子问题1, 子问题2]
                               ↓
                          LLM → [子子问题1a, 子子问题1b]
                               ↓
                          LLM → 基础答案
                               ↓
                     合并 → 最终答案

典型应用场景：
  - 复杂推理：数学证明、多步逻辑
  - 层级摘要：先摘要段落，再摘要章节，再摘要全书
  - 树状任务分解：大任务 → 子任务 → 子子任务
  - 代码生成：先设计架构 → 再生成模块 → 再生成函数
"""

import os
from dotenv import load_dotenv
from openai import OpenAI

# 加载环境变量（复用 advanced-rag 的 .env）
load_dotenv(dotenv_path="../advanced-rag/.env")

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
)

# ─────────────────────────────────────────────
# 核心：RLM 递归调用器
# ─────────────────────────────────────────────

def llm_call(prompt: str, depth: int = 0) -> str:
    """单次 LLM 调用，带缩进显示层级"""
    indent = "  " * depth
    print(f"{indent}🤖 [LLM 调用 depth={depth}] {prompt[:60]}...")
    
    response = client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    result = response.choices[0].message.content.strip()
    print(f"{indent}   ↩ {result[:80]}...")
    return result


def rlm_solve(problem: str, depth: int = 0, max_depth: int = 3) -> str:
    """
    RLM 核心递归函数
    
    流程：
    1. 让 LLM 判断问题是否可以直接回答（base case）
    2. 如果不能，让 LLM 拆解成子问题
    3. 递归解决每个子问题
    4. 让 LLM 合并所有子答案得出最终答案
    
    Args:
        problem: 要解决的问题
        depth: 当前递归深度
        max_depth: 最大递归深度（防止无限递归）
    
    Returns:
        问题的最终答案
    """
    indent = "  " * depth
    print(f"\n{indent}{'='*50}")
    print(f"{indent}📌 [depth={depth}] 处理问题: {problem}")
    
    # ── Base Case：超过最大深度，强制直接回答 ──
    if depth >= max_depth:
        print(f"{indent}⚠️  达到最大深度，强制直接回答")
        return llm_call(
            f"请直接简短回答这个问题（不要再拆解）：{problem}",
            depth=depth
        )
    
    # ── Step 1：判断是否可以直接回答 ──
    can_answer_prompt = f"""判断以下问题是否可以直接回答，不需要拆解成子问题。
    
问题：{problem}

如果可以直接回答，回复：DIRECT
如果需要拆解，回复：DECOMPOSE
只回复一个词，不要解释。"""
    
    decision = llm_call(can_answer_prompt, depth=depth)
    
    # ── Base Case：可以直接回答 ──
    if "DIRECT" in decision.upper():
        print(f"{indent}✅ 直接回答")
        answer = llm_call(f"请回答：{problem}", depth=depth)
        return answer
    
    # ── Recursive Case：需要拆解 ──
    print(f"{indent}🔀 需要拆解，进入递归...")
    
    # Step 2：拆解成子问题
    decompose_prompt = f"""将以下复杂问题拆解成 2-3 个独立的子问题。

问题：{problem}

要求：
- 每个子问题独立可解
- 子问题合并后能回答原问题
- 每行一个子问题，用数字编号

只输出子问题列表，格式：
1. 子问题一
2. 子问题二
3. 子问题三（可选）"""
    
    decomposition = llm_call(decompose_prompt, depth=depth)
    
    # 解析子问题
    sub_problems = []
    for line in decomposition.strip().split("\n"):
        line = line.strip()
        if line and line[0].isdigit():
            # 去掉编号，提取问题内容
            sub_problem = line.split(".", 1)[-1].strip()
            if sub_problem:
                sub_problems.append(sub_problem)
    
    print(f"{indent}📋 拆解出 {len(sub_problems)} 个子问题:")
    for i, sp in enumerate(sub_problems, 1):
        print(f"{indent}   {i}. {sp}")
    
    # Step 3：递归解决每个子问题
    sub_answers = {}
    for i, sub_problem in enumerate(sub_problems, 1):
        print(f"\n{indent}→ 解决子问题 {i}/{len(sub_problems)}")
        sub_answer = rlm_solve(sub_problem, depth=depth+1, max_depth=max_depth)
        sub_answers[sub_problem] = sub_answer
    
    # Step 4：合并子答案
    print(f"\n{indent}🔗 合并子答案...")
    
    sub_qa_text = "\n".join([
        f"子问题：{q}\n子答案：{a}" 
        for q, a in sub_answers.items()
    ])
    
    merge_prompt = f"""基于以下子问题和子答案，综合回答原始问题。

原始问题：{problem}

子问题解答：
{sub_qa_text}

请综合以上信息，给出原始问题的完整答案："""
    
    final_answer = llm_call(merge_prompt, depth=depth)
    
    print(f"{indent}✨ [depth={depth}] 最终答案: {final_answer[:100]}...")
    return final_answer


# ─────────────────────────────────────────────
# 示例 1：简单问题（不需要递归，直接回答）
# ─────────────────────────────────────────────

def demo_simple():
    print("\n" + "="*60)
    print("示例 1：简单问题（base case，直接回答）")
    print("="*60)
    
    question = "Python 的 list 和 tuple 有什么区别？"
    answer = rlm_solve(question, max_depth=2)
    
    print(f"\n📝 最终答案:\n{answer}")


# ─────────────────────────────────────────────
# 示例 2：复杂问题（需要递归拆解）
# ─────────────────────────────────────────────

def demo_complex():
    print("\n" + "="*60)
    print("示例 2：复杂问题（需要递归拆解）")
    print("="*60)
    
    question = "如何设计一个高并发的电商秒杀系统？"
    answer = rlm_solve(question, max_depth=2)
    
    print(f"\n📝 最终答案:\n{answer}")


# ─────────────────────────────────────────────
# 示例 3：层级摘要（RLM 的经典应用）
# ─────────────────────────────────────────────

def demo_hierarchical_summary():
    """
    层级摘要：先摘要每个段落，再合并成整体摘要
    这是 RLM 最典型的实际应用——处理超出 context window 的长文档
    """
    print("\n" + "="*60)
    print("示例 3：层级摘要（RLM 处理长文档）")
    print("="*60)
    
    # 模拟一篇长文档的三个章节
    chapters = [
        "第一章：RAG（检索增强生成）通过在 LLM 生成前先检索相关文档，解决了 LLM 知识截止和幻觉问题。核心组件包括向量数据库、Embedding 模型和检索器。",
        "第二章：Agent 是能够使用工具、规划行动并自主完成任务的 LLM 应用。ReAct 框架通过 Thought-Action-Observation 循环实现了稳定的推理和行动交替。",
        "第三章：Multi-Agent 系统让多个专门化的 Agent 协作完成复杂任务。常见模式包括 Orchestrator-Worker、Swarm 和 Pipeline，各有不同的控制权流转机制。",
    ]
    
    print("📚 原始文档（3个章节）")
    
    # 第一层递归：摘要每个章节
    chapter_summaries = []
    for i, chapter in enumerate(chapters, 1):
        print(f"\n→ 摘要第 {i} 章...")
        summary = llm_call(
            f"用一句话摘要以下内容：{chapter}",
            depth=1
        )
        chapter_summaries.append(summary)
    
    # 第二层递归：合并章节摘要为整体摘要
    print(f"\n→ 合并所有章节摘要...")
    combined = "\n".join([f"第{i+1}章摘要：{s}" for i, s in enumerate(chapter_summaries)])
    final_summary = llm_call(
        f"基于以下章节摘要，写一段整体摘要：\n{combined}",
        depth=0
    )
    
    print(f"\n📝 整体摘要:\n{final_summary}")


# ─────────────────────────────────────────────
# 主入口
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("🔄 RLM (Recursive Language Model) Demo")
    print("递归语言模型：让 LLM 递归调用自身解决复杂问题\n")
    
    # 运行示例（按需注释/取消注释）
    
    # 示例1：简单问题，直接回答（不递归）
    demo_simple()
    
    # 示例2：复杂问题，自动拆解（会递归）
    # demo_complex()
    
    # 示例3：层级摘要（手动实现 RLM 的经典应用）
    # demo_hierarchical_summary()
