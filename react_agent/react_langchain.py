"""
ReAct Agent - LangGraph 框架版（文本格式驱动，兼容不支持 function calling 的模型）

LangChain 1.x / LangGraph 默认用 function calling 来驱动工具调用。
但部分模型（如 Qwen3 reasoning 模式）不支持 function calling，需要改用文本格式。

本文件演示：用 LangGraph 构建 StateGraph，手动实现工具路由，
兼容不支持 function calling 的模型，同时保留 LangGraph 的图结构优势。

重点观察：
  - LangGraph 用节点（Node）和边（Edge）描述 Agent 流程
  - llm_node：调用 LLM，解析输出决定下一步
  - tool_node：执行工具，返回 Observation
  - 边：根据 LLM 输出决定走 tool_node 还是结束

运行方式：
  python react_langchain.py
"""

import os
import json
import re
from typing import TypedDict, Annotated
from dotenv import load_dotenv

if os.path.exists(".env"):
    load_dotenv(".env")
elif os.path.exists("../advanced-rag/.env"):
    load_dotenv("../advanced-rag/.env")

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages


# ============================================================
# 工具定义（普通函数，不用 @tool 装饰器）
# ============================================================

def get_company_info(company_name: str) -> str:
    mock_data = {
        "比亚迪": {"industry": "新能源汽车", "employees": "约70万人", "market_position": "全球新能源汽车销量第一"},
        "宁德时代": {"industry": "动力电池", "employees": "约10万人", "market_position": "全球动力电池市占率第一"},
    }
    return json.dumps(mock_data.get(company_name, {"error": "未找到"}), ensure_ascii=False)

def get_financial_data(company_name: str) -> str:
    mock_data = {
        "比亚迪": {"revenue": "6023亿元", "revenue_growth": "+42%", "net_profit": "300亿元", "pe_ratio": "约25倍"},
        "宁德时代": {"revenue": "4009亿元", "revenue_growth": "+22%", "net_profit": "441亿元", "pe_ratio": "约20倍"},
    }
    return json.dumps(mock_data.get(company_name, {"error": "未找到"}), ensure_ascii=False)

def get_news_sentiment(company_name: str) -> str:
    mock_data = {
        "比亚迪": {"sentiment": "偏正面", "risks": ["欧盟关税", "价格战"], "news": ["销量创新高", "进军欧洲"]},
        "宁德时代": {"sentiment": "中性偏正面", "risks": ["客户自研电池", "锂价波动"], "news": ["固态电池突破", "海外建厂"]},
    }
    return json.dumps(mock_data.get(company_name, {"error": "未找到"}), ensure_ascii=False)

def get_competitor_analysis(company_name: str) -> str:
    mock_data = {
        "比亚迪": {"advantages": ["垂直整合", "成本控制"], "weaknesses": ["品牌高端化", "海外刚起步"]},
        "宁德时代": {"advantages": ["技术领先", "客户绑定深"], "weaknesses": ["客户自研风险", "折旧压力"]},
    }
    return json.dumps(mock_data.get(company_name, {"error": "未找到"}), ensure_ascii=False)

# 工具注册表
TOOLS = {
    "get_company_info": get_company_info,
    "get_financial_data": get_financial_data,
    "get_news_sentiment": get_news_sentiment,
    "get_competitor_analysis": get_competitor_analysis,
}

TOOL_DESCRIPTIONS = """
- get_company_info(company_name): 获取公司行业、规模、市场地位
- get_financial_data(company_name): 获取营收、利润、增长率、PE估值
- get_news_sentiment(company_name): 获取近期新闻舆情和风险点
- get_competitor_analysis(company_name): 获取竞争优势和劣势分析
"""

SYSTEM_PROMPT = f"""你是专业的股票研究分析师。

可用工具：{TOOL_DESCRIPTIONS}

严格按以下格式输出（每次只输出一个步骤）：
Thought: [你的推理]
Action: [工具名]
Action Input: {{"参数名": "参数值"}}

当信息足够时（至少调用3次工具后），输出：
Thought: [总结性推理]
Final Answer: [完整的投资分析报告，包含推荐/中性/谨慎的明确建议]
"""


# ============================================================
# LangGraph State 定义
# ============================================================

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]  # 消息历史，自动追加
    scratchpad: str                           # ReAct scratchpad
    step_count: int                           # 步骤计数
    finished: bool                            # 是否完成


# ============================================================
# LangGraph 节点定义
# ============================================================

llm = ChatOpenAI(
    model=os.getenv("LLM_MODEL", "gpt-4o"),
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
    temperature=0,
    max_tokens=2000,
)

def llm_node(state: AgentState) -> AgentState:
    """LLM 节点：根据当前 scratchpad 决定下一步（Action 或 Final Answer）"""
    task = state["messages"][0].content  # 原始任务
    scratchpad = state.get("scratchpad", "")
    step = state.get("step_count", 0) + 1

    print(f"\n{'─' * 60}")
    print(f"⚙️  步骤 {step} - LLM 节点")

    # 构建 prompt：任务 + scratchpad 历史
    user_content = task
    if scratchpad:
        user_content += f"\n\n{scratchpad.strip()}"

    response = llm.invoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_content),
    ])

    output = response.content.strip()

    # 解析输出
    thought_match = re.search(r"Thought:\s*(.+?)(?=Action:|Final Answer:|$)", output, re.DOTALL)
    thought = thought_match.group(1).strip() if thought_match else ""

    final_match = re.search(r"Final Answer:\s*(.+)", output, re.DOTALL)
    action_match = re.search(r"Action:\s*(\w+)", output)
    input_match = re.search(r"Action Input:\s*(\{.+?\})", output, re.DOTALL)

    if thought:
        print(f"💭 Thought: {thought[:150]}")

    if final_match:
        # 最终答案
        final = final_match.group(1).strip()
        print(f"✅ Final Answer 生成完毕")
        return {
            **state,
            "messages": [AIMessage(content=final)],
            "finished": True,
            "step_count": step,
        }
    elif action_match:
        # 需要执行工具
        action = action_match.group(1).strip()
        action_input = {}
        if input_match:
            try:
                action_input = json.loads(input_match.group(1))
            except Exception:
                pass

        print(f"🔧 Action: {action}({json.dumps(action_input, ensure_ascii=False)})")

        # 把这一步的 Thought+Action 追加到 scratchpad
        new_scratchpad = scratchpad + f"\nThought: {thought}\nAction: {action}\nAction Input: {json.dumps(action_input, ensure_ascii=False)}\n"

        return {
            **state,
            "scratchpad": new_scratchpad,
            "step_count": step,
            "_pending_action": action,
            "_pending_input": action_input,
        }
    else:
        print(f"⚠️  格式异常，原始输出: {output[:200]}")
        return {**state, "finished": True, "step_count": step}


def tool_node(state: AgentState) -> AgentState:
    """工具节点：执行上一步 LLM 决定的工具调用，得到 Observation"""
    action = state.get("_pending_action")
    action_input = state.get("_pending_input", {})

    if action and action in TOOLS:
        try:
            observation = TOOLS[action](**action_input)
        except Exception as e:
            observation = f"工具执行出错: {e}"
    else:
        observation = f"工具 {action} 不存在"

    print(f"👁️  Observation: {observation[:200]}")

    # 把 Observation 追加到 scratchpad
    new_scratchpad = state.get("scratchpad", "") + f"Observation: {observation}\n"

    return {
        **state,
        "scratchpad": new_scratchpad,
        "_pending_action": None,
        "_pending_input": None,
    }


def should_continue(state: AgentState) -> str:
    """路由函数：决定下一个节点是 tool_node 还是 END"""
    if state.get("finished"):
        return END
    if state.get("step_count", 0) >= 10:  # 防止无限循环
        return END
    if state.get("_pending_action"):
        return "tool_node"
    return "llm_node"


# ============================================================
# 构建 LangGraph 图
# ============================================================

def build_agent_graph():
    graph = StateGraph(AgentState)

    # 添加节点
    graph.add_node("llm_node", llm_node)
    graph.add_node("tool_node", tool_node)

    # 设置入口
    graph.set_entry_point("llm_node")

    # 条件边：llm_node 执行后，根据 should_continue 决定走哪里
    graph.add_conditional_edges(
        "llm_node",
        should_continue,
        {
            "tool_node": "tool_node",
            "llm_node": "llm_node",
            END: END,
        }
    )

    # tool_node 执行后固定回到 llm_node
    graph.add_edge("tool_node", "llm_node")

    return graph.compile()


# ============================================================
# 主程序
# ============================================================

def main():
    agent = build_agent_graph()

    task = "请分析比亚迪公司是否值得长期投资，需要从基本面、财务、舆情、竞争四个维度分析"

    print("=" * 70)
    print("ReAct Agent - LangGraph 图结构版（文本格式，兼容 reasoning 模型）")
    print("=" * 70)
    print(f"\n📋 任务: {task}")
    print("\n图结构：llm_node → tool_node → llm_node → ... → END")
    print("─" * 70)

    initial_state = {
        "messages": [HumanMessage(content=task)],
        "scratchpad": "",
        "step_count": 0,
        "finished": False,
        "_pending_action": None,
        "_pending_input": None,
    }

    result = agent.invoke(initial_state)

    # 提取最终答案
    final_answer = ""
    for msg in reversed(result["messages"]):
        if isinstance(msg, AIMessage) and msg.content:
            final_answer = msg.content
            break

    print("\n" + "=" * 70)
    print("✅ Final Answer:")
    print("─" * 70)
    print(final_answer)
    print("─" * 70)
    print(f"\n总步骤数: {result['step_count']}")
    print("\n框架 vs 手写版 核心差异：")
    print("  手写版: for 循环 + 字符串拼接")
    print("  LangGraph: 有向图 + 节点/边 + 状态机")
    print("  优势: 图结构更容易扩展（加节点）、可视化、并行执行")


if __name__ == "__main__":
    main()
