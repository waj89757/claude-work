"""
DeepAgents Demo - 竞品分析 Agent
====================================
演示 deepagents 的核心能力：
  1. 多工具调用（行业数据、竞品对比、新闻检索）
  2. 子 Agent（专职研究员，context 隔离）
  3. Human-in-the-loop（写报告前需要人工审批）
  4. 自定义 Middleware（记录每次工具调用）
  5. 流式输出

场景：你是一个产品经理，想分析「抖音 vs 快手」的竞争态势，
     Agent 自动拆解任务 → 调工具 → 委托子 Agent 深调研 → 人工审批 → 生成报告

运行方式：
  pip install deepagents langchain-openai langgraph
  python deep_agent_demo.py
"""

import os
import sys

# ──────────────────────────────────────────────
# 1. 加载环境变量（复用 advanced-rag/.env）
# ──────────────────────────────────────────────
from pathlib import Path
from dotenv import load_dotenv

env_path = Path(__file__).parent.parent / "advanced-rag" / ".env"
load_dotenv(env_path)

API_KEY    = os.getenv("OPENAI_API_KEY")
BASE_URL   = os.getenv("OPENAI_BASE_URL")
MODEL_NAME = os.getenv("LLM_MODEL")

if not all([API_KEY, BASE_URL, MODEL_NAME]):
    print("❌ 缺少环境变量，请检查 advanced-rag/.env")
    sys.exit(1)

print(f"✅ 已加载配置: model={MODEL_NAME}")
print(f"   base_url={BASE_URL}\n")

# ──────────────────────────────────────────────
# 2. 创建 LangChain ChatOpenAI 模型对象
#    (deepagents 支持传入 ChatModel 实例，不限于 "provider:model" 字符串)
# ──────────────────────────────────────────────
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model=MODEL_NAME,
    api_key=API_KEY,
    base_url=BASE_URL,
    temperature=0,
    max_tokens=4000,
)

# ──────────────────────────────────────────────
# 3. 定义工具（模拟真实数据源）
# ──────────────────────────────────────────────
from langchain.tools import tool


@tool
def get_platform_dau(platform: str) -> str:
    """获取平台的日活跃用户数(DAU)和月活跃用户数(MAU)等核心指标。
    platform: 平台名称，如 '抖音', '快手', 'B站' 等
    """
    data = {
        "抖音": {
            "DAU": "7.5亿",
            "MAU": "11亿",
            "人均使用时长": "120分钟/天",
            "用户画像": "18-35岁城市用户为主，女性占比55%",
            "商业化收入": "2400亿元（2024年）",
        },
        "快手": {
            "DAU": "3.9亿",
            "MAU": "7亿",
            "人均使用时长": "115分钟/天",
            "用户画像": "下沉市场和农村用户，男性占比55%，30+用户更多",
            "商业化收入": "1140亿元（2024年）",
        },
        "B站": {
            "DAU": "1.03亿",
            "MAU": "3.3亿",
            "人均使用时长": "100分钟/天",
            "用户画像": "18-25岁年轻用户，Z世代聚集地",
            "商业化收入": "225亿元（2024年）",
        },
    }
    if platform in data:
        metrics = data[platform]
        return f"【{platform} 核心指标】\n" + "\n".join(
            f"  - {k}: {v}" for k, v in metrics.items()
        )
    return f"暂无 {platform} 的数据，已知平台：{', '.join(data.keys())}"


@tool
def get_revenue_breakdown(platform: str) -> str:
    """获取平台收入结构拆解（广告、电商、直播打赏等）。
    platform: 平台名称
    """
    data = {
        "抖音": {
            "广告收入": "1500亿（占比62%）",
            "电商抽佣": "600亿（占比25%）",
            "直播打赏": "250亿（占比10%）",
            "其他": "50亿（占比3%）",
        },
        "快手": {
            "广告收入": "450亿（占比39%）",
            "直播打赏": "520亿（占比46%）",
            "电商抽佣": "140亿（占比12%）",
            "其他": "30亿（占比3%）",
        },
    }
    if platform in data:
        breakdown = data[platform]
        return f"【{platform} 收入结构】\n" + "\n".join(
            f"  - {k}: {v}" for k, v in breakdown.items()
        )
    return f"暂无 {platform} 的收入结构数据"


@tool
def get_recent_strategy(platform: str) -> str:
    """获取平台近期战略动向和重大事件。
    platform: 平台名称
    """
    data = {
        "抖音": """近期战略动向：
  1. 全力发力"货架电商"，推进抖音商城独立App，对标淘宝天猫
  2. 出海策略：TikTok在印度被封后，重点布局东南亚和拉美市场
  3. AI基础设施：豆包大模型开放API，打造AI创作生态
  4. 本地生活：与美团正面竞争，抖音到店/到家业务快速增长""",
        "快手": """近期战略动向：
  1. 聚焦"信任电商"差异化，主播与粉丝深度连接是核心壁垒
  2. 出海：Kwai在巴西月活超1亿，成为重要增量市场
  3. 磁力引擎升级，广告技术栈迭代，追赶穿山甲（字节广告）
  4. AI方向：可灵视频生成模型，在AI视频赛道取得领先""",
    }
    return data.get(platform, f"暂无 {platform} 的战略数据")


@tool
def compare_feature_matrix(feature: str) -> str:
    """对比两个平台在某一具体功能或维度的差异。
    feature: 对比维度，如 '推荐算法', '电商能力', 'AI创作工具', '创作者激励' 等
    """
    matrix = {
        "推荐算法": """推荐算法对比：
  抖音：纯兴趣图谱，去社交化，完播率/点赞/评论驱动，冷启动对新人友好
  快手：双列信息流 + 关注权重更高，社交图谱占比大，老用户黏性更强
  核心差异：抖音让用户沉迷刷视频，快手让用户关注人""",
        "电商能力": """电商能力对比：
  抖音：GMV约3万亿（2024），货架+直播双轮驱动，品牌商家更多
  快手：GMV约1.2万亿（2024），信任电商模型，下沉市场农产品/白牌强
  核心差异：抖音品牌化，快手私域复购率更高""",
        "AI创作工具": """AI创作工具对比：
  抖音：剪映（专业级）+ 即梦（图文生成），AI配音/字幕/特效成熟
  快手：可灵（视频生成全球领先）+ 快影（轻量），可灵2.0对标Sora
  核心差异：快手在AI视频生成领域暂时领先，抖音在编辑工具生态更完善""",
        "创作者激励": """创作者激励对比：
  抖音：中视频计划（1000粉+）+ 直播分成，流量导入强但平台抽成也高
  快手：光合计划 + 磁力聚星，更倾向中腰部达人，分成比例更友好
  核心差异：快手对中小创作者更友善，抖音对头部达人更有利""",
    }
    return matrix.get(feature, f"暂不支持对比维度 '{feature}'，可选：{', '.join(matrix.keys())}")


@tool
def write_competitive_report(
    title: str,
    executive_summary: str,
    key_findings: str,
    recommendations: str,
) -> str:
    """将分析结果整理成正式的竞品分析报告并保存。
    title: 报告标题
    executive_summary: 执行摘要（3-5句话）
    key_findings: 主要发现（Markdown格式）
    recommendations: 战略建议（Markdown格式）
    """
    report = f"""
# {title}

---

## 执行摘要

{executive_summary}

---

## 主要发现

{key_findings}

---

## 战略建议

{recommendations}

---

*报告生成时间：{__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M')}*
*数据来源：内部分析工具*
"""
    # 保存到文件
    output_path = Path(__file__).parent / "competitive_report.md"
    output_path.write_text(report, encoding="utf-8")
    return f"✅ 报告已保存至 {output_path}\n\n报告预览（前500字）：\n{report[:500]}..."


# ──────────────────────────────────────────────
# 4. 自定义 Middleware：完整记录 LLM 输入输出 + 工具调用
# ──────────────────────────────────────────────
from langchain.agents.middleware import AgentMiddleware

_llm_call_count = 0  # 模块级计数器（只用于打印，不影响并发安全）


class ToolCallLoggerMiddleware(AgentMiddleware):
    """完整记录每次 LLM 调用的输入/输出，以及每次工具调用的参数和返回值。"""

    def wrap_model_call(self, request, handler):
        """拦截 LLM 调用：打印发送的 messages（输入）和 AI 返回的 message（输出）。

        request.state.messages  → 当前完整消息历史（发给 LLM 的 context）
        response.result         → LLM 返回的 AIMessage 列表
        """
        global _llm_call_count
        _llm_call_count += 1
        call_no = _llm_call_count

        # ── 打印 LLM 输入 ──
        state = request.state if hasattr(request, 'state') else {}
        # state 可能是 dict 或对象，兼容两种情况
        if isinstance(state, dict):
            messages = state.get("messages", [])
        else:
            messages = getattr(state, "messages", [])
        print(f"\n{'='*60}")
        print(f"🧠 [LLM 调用 #{call_no}] 发送 {len(messages)} 条消息")
        print(f"{'='*60}")
        for i, msg in enumerate(messages):
            role = getattr(msg, 'type', type(msg).__name__)
            content = getattr(msg, 'content', str(msg))
            # content 可能是 list（多模态）或 str
            if isinstance(content, list):
                content_str = str(content)[:200]
            else:
                content_str = str(content)[:300]
            if len(str(getattr(msg, 'content', ''))) > 300:
                content_str += "...[截断]"
            # 工具调用信息
            tool_calls = getattr(msg, 'tool_calls', [])
            print(f"  [{i+1}] {role}: {content_str}")
            if tool_calls:
                for tc in tool_calls:
                    tc_name = tc.get('name', '?') if isinstance(tc, dict) else getattr(tc, 'name', '?')
                    print(f"       └─ tool_call: {tc_name}")
        print(f"  {'─'*50}")

        # ── 实际调用 LLM ──
        response = handler(request)

        # ── 打印 LLM 输出 ──
        results = response.result if hasattr(response, 'result') else [response]
        for ai_msg in results:
            content = getattr(ai_msg, 'content', '')
            tool_calls = getattr(ai_msg, 'tool_calls', [])
            content_preview = str(content)[:400] + "...[截断]" if len(str(content)) > 400 else str(content)
            print(f"  🤖 AI 回复: {content_preview}")
            if tool_calls:
                print(f"  📌 决定调用工具:")
                for tc in tool_calls:
                    tc_name = tc.get('name', '?') if isinstance(tc, dict) else getattr(tc, 'name', '?')
                    tc_args = tc.get('args', {}) if isinstance(tc, dict) else getattr(tc, 'args', {})
                    print(f"     ├─ {tc_name}({tc_args})")
            else:
                print(f"  ✅ 无工具调用（本轮直接回答）")
        print(f"{'='*60}")

        return response

    def wrap_tool_call(self, request, handler):
        """拦截工具调用：打印工具名、参数和返回值。"""
        tool_name = request.tool_call.get("name", "unknown")
        tool_args = request.tool_call.get("args", {})

        print(f"\n🔧 [执行工具] {tool_name}")
        if tool_args:
            for k, v in tool_args.items():
                val_str = str(v)[:150] + "..." if len(str(v)) > 150 else str(v)
                print(f"   ├─ {k}: {val_str}")

        result = handler(request)

        result_content = getattr(result, "content", str(result))
        preview = str(result_content)[:200] + "..." if len(str(result_content)) > 200 else str(result_content)
        print(f"   └─ 返回: {preview}")

        return result


# ──────────────────────────────────────────────
# 5. 构建 DeepAgent
# ──────────────────────────────────────────────
from deepagents import create_deep_agent
from langgraph.checkpoint.memory import MemorySaver

# Human-in-the-loop 需要 checkpointer
checkpointer = MemorySaver()

# 定义子 Agent：专职深度研究员
# 当主 Agent 觉得某个问题需要更深入调研时，会把任务委托给它
# 子 Agent 有独立的 context window，不会污染主 Agent 的对话历史
research_subagent = {
    "name": "deep-researcher",
    "description": "专门用于深度调研某一具体问题，如竞品某项功能的详细分析、行业趋势深挖等。输入需要调研的具体问题，返回详细的调研报告。",
    "system_prompt": """你是一名专业的互联网行业分析师。
当你被委托调研某个问题时：
1. 首先明确调研目标
2. 使用所有可用工具收集数据
3. 从多个维度进行分析
4. 给出有洞察力的结论，不要只是罗列数据

输出格式要结构化，方便主Agent整合到报告中。""",
    "tools": [get_platform_dau, get_revenue_breakdown, compare_feature_matrix],
    # 不指定 model，默认继承主 Agent 的模型
}

# 创建主 Agent
agent = create_deep_agent(
    model=llm,  # 传入 ChatModel 实例（兼容 OpenAI 接口）
    tools=[
        get_platform_dau,
        get_revenue_breakdown,
        get_recent_strategy,
        compare_feature_matrix,
        write_competitive_report,  # ⚠️ 这个工具会触发 human-in-the-loop
    ],
    system_prompt="""你是一名专业的产品竞品分析师。
你的工作流程：
1. 先拆解任务，列出 todo
2. 系统收集两个平台的核心数据（DAU/MAU、收入结构、战略动向）
3. 对关键维度进行横向对比
4. 如果需要深入某个话题，委托 deep-researcher 子Agent进行专项调研
5. 综合所有信息，调用 write_competitive_report 生成正式报告（注意：此步骤需要人工审批）

分析要有洞察力，不要只是数据堆砌。要指出：
- 双方真正的差异化在哪里
- 各自的核心壁垒
- 竞争格局的走向判断""",
    subagents=[research_subagent],
    middleware=[ToolCallLoggerMiddleware()],
    # Human-in-the-loop：write_competitive_report 调用前需要人工确认
    interrupt_on={
        "write_competitive_report": {
            "allowed_decisions": ["approve", "reject"],  # 只允许通过或拒绝，不允许修改参数
        },
    },
    checkpointer=checkpointer,
)

# ──────────────────────────────────────────────
# 6. 运行 Agent（带 Human-in-the-loop 处理）
# ──────────────────────────────────────────────
print("=" * 60)
print("🚀 DeepAgents 竞品分析 Demo")
print("=" * 60)
print()
print("任务：分析抖音 vs 快手的竞争态势，生成竞品分析报告")
print()

# thread_id 用于 checkpointer 跟踪会话状态（human-in-the-loop 必须）
thread_config = {"configurable": {"thread_id": "competitive-analysis-001"}}

task = {
    "messages": [
        {
            "role": "user",
            "content": """请帮我做一份「抖音 vs 快手」的竞品分析报告。

需要包含：
1. 核心规模指标对比（DAU、MAU、使用时长）
2. 商业化收入结构对比
3. 近期战略动向
4. 推荐算法 + 电商能力 + AI创作工具的功能对比
5. 综合判断：快手的差异化机会在哪里？

最后生成一份正式的竞品分析报告文件。""",
        }
    ]
}

# 第一阶段：Agent 自主运行，直到触发 human-in-the-loop 暂停
print("📋 开始执行任务（Agent 自主运行中...）\n")

try:
    # stream 模式：实时看到 Agent 的每一步
    interrupted = False
    interrupt_data = None

    for chunk in agent.stream(task, config=thread_config):
        # ── 打印 Graph 节点切换信息 ──
        for node_name, node_output in chunk.items():
            if node_name == "__interrupt__":
                continue
            # 只打印节点名，让流程可见
            msgs = node_output.get("messages", []) if isinstance(node_output, dict) else []
            print(f"\n▶ [节点] {node_name}  ({len(msgs)} 条消息)")

        # 检查是否触发了中断（human-in-the-loop）
        if "__interrupt__" in chunk:
            interrupts = chunk["__interrupt__"]
            if interrupts:
                interrupted = True
                interrupt_data = interrupts[0]
                break

    if interrupted and interrupt_data:
        # ── Human-in-the-loop 处理 ──
        print("\n" + "=" * 60)
        print("⚠️  [Human-in-the-Loop] Agent 请求人工审批")
        print("=" * 60)
        print("\nAgent 即将调用 write_competitive_report 工具生成报告。")

        # 显示 Agent 准备写入的报告参数
        if hasattr(interrupt_data, "value") and isinstance(interrupt_data.value, dict):
            tool_call = interrupt_data.value.get("tool_call", {})
            args = tool_call.get("args", {})
            if args:
                print("\n📄 报告预览：")
                print(f"  标题: {args.get('title', 'N/A')}")
                print(f"  摘要: {args.get('executive_summary', 'N/A')[:200]}...")

        print("\n请选择操作：")
        print("  [a] approve  - 批准，生成报告")
        print("  [r] reject   - 拒绝，不生成报告")

        while True:
            choice = input("\n你的决定 (a/r): ").strip().lower()
            if choice in ["a", "approve"]:
                decision = "approve"
                break
            elif choice in ["r", "reject"]:
                decision = "reject"
                break
            else:
                print("请输入 a 或 r")

        print(f"\n✅ 你的决定：{decision}")

        # 恢复执行
        from langgraph.types import Command

        print("\n📋 继续执行（处理审批结果中...）\n")
        for chunk in agent.stream(
            Command(resume={"decisions": [{"type": decision}]}),
            config=thread_config,
        ):
            for node_name, node_output in chunk.items():
                if node_name == "__interrupt__":
                    continue
                msgs = node_output.get("messages", []) if isinstance(node_output, dict) else []
                print(f"\n▶ [节点] {node_name}  ({len(msgs)} 条消息)")

        print("\n✅ 任务执行完成！")

    else:
        print("\n✅ 任务执行完成（无需人工审批）")

except KeyboardInterrupt:
    print("\n\n👋 用户中断执行")
except Exception as e:
    print(f"\n❌ 执行出错: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

# ──────────────────────────────────────────────
# 7. 显示最终结果
# ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("📊 执行总结")
print("=" * 60)

report_path = Path(__file__).parent / "competitive_report.md"
if report_path.exists():
    print(f"\n✅ 报告已生成：{report_path}")
    content = report_path.read_text(encoding="utf-8")
    print(f"\n--- 报告内容预览（前1000字）---\n{content[:1000]}")
else:
    print("\n⚠️  报告未生成（可能被拒绝或执行未完成）")
