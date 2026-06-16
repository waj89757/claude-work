"""
Demo 3: Swarm - 控制权移交模式

场景：Planner 接收任务后，自主决定分配给 Researcher 或 Writer
特点：
  - 任意时刻只有一个 Agent 是"Active Speaker"（持有控制权）
  - Agent 通过发出 HandoffMessage 把控制权显式移交给下一个
  - 第一个持有控制权的 Agent 由 participants[0] 决定（开发者指定）
  - 之后完全由 Agent 自主决定移交给谁

关键观察：
  - 看 HandoffMessage 的出现时机，理解控制权流转
  - Planner 不会在 Researcher 和 Writer 工作时插嘴
  - 如果 Agent 忘了移交，流程就卡在那里（这是 Swarm 的风险）
  
控制权流转图：
  用户任务 → Planner → 移交给 Researcher → 移交给 Writer → 移交给 Planner → 结束
"""

import asyncio
import os
from dotenv import load_dotenv

if os.path.exists(".env"):
    load_dotenv(".env")
elif os.path.exists("../advanced-rag/.env"):
    load_dotenv("../advanced-rag/.env")
    print("使用 advanced-rag/.env 配置\n")

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import Swarm
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.messages import HandoffMessage
from autogen_ext.models.openai import OpenAIChatCompletionClient


async def main():
    api_key = os.getenv("OPENAI_API_KEY", "sk-placeholder")
    base_url = os.getenv("OPENAI_BASE_URL")
    model = os.getenv("LLM_MODEL", "gpt-4o")

    print(f"使用模型: {model}")
    print(f"Base URL: {base_url}")
    print("=" * 60)

    model_client = OpenAIChatCompletionClient(
        model=model,
        api_key=api_key,
        base_url=base_url,
        model_capabilities={
            "vision": False,
            "function_calling": True,
            "json_output": True,
        },
    )

    # Planner：持有初始控制权，负责拆解任务并决定移交给谁
    # 注意 system_message 里必须明确告知"完成后移交给谁"
    planner = AssistantAgent(
        name="Planner",
        model_client=model_client,
        handoffs=["Researcher", "Writer"],  # 声明可以移交给哪些 Agent
        system_message=(
            "你是项目规划者，你是第一个接收任务的人（初始控制权持有者）。\n"
            "职责：\n"
            "1. 分析任务，制定研究提纲（3个要点）\n"
            "2. 完成提纲后，把控制权移交给 Researcher 去收集信息\n"
            "3. 当 Writer 完成写作后，你会收回控制权，做最终质量检查\n"
            "4. 确认质量OK后，输出 'TERMINATE' 结束任务\n\n"
            "重要：完成提纲后，必须通过 handoff 把控制权传给 Researcher。"
        ),
    )

    # Researcher：负责信息调研，完成后移交给 Writer
    researcher = AssistantAgent(
        name="Researcher",
        model_client=model_client,
        handoffs=["Writer", "Planner"],  # 可以移交给 Writer 或退回给 Planner
        system_message=(
            "你是信息研究员，你在收到 Planner 移交的控制权后开始工作。\n"
            "职责：\n"
            "1. 根据 Planner 的提纲，提供每个要点的详细信息和数据（简短，每点50字）\n"
            "2. 完成研究后，把控制权移交给 Writer 去写作\n"
            "3. 如果提纲不清晰，退回控制权给 Planner\n\n"
            "重要：完成研究后，必须通过 handoff 把控制权传给 Writer。"
        ),
    )

    # Writer：负责写作，完成后移交回 Planner 做最终确认
    writer = AssistantAgent(
        name="Writer",
        model_client=model_client,
        handoffs=["Planner"],  # 只能移交回给 Planner
        system_message=(
            "你是内容撰写者，你在收到 Researcher 移交的控制权后开始工作。\n"
            "职责：\n"
            "1. 基于 Researcher 提供的信息，写出结构清晰的内容（200字左右）\n"
            "2. 完成写作后，把控制权移交回 Planner 做最终确认\n\n"
            "重要：完成写作后，必须通过 handoff 把控制权传给 Planner。"
        ),
    )

    # Swarm：participants[0] = Planner = 初始控制权持有者
    team = Swarm(
        participants=[planner, researcher, writer],  # planner 排第一，拿到初始控制权
        termination_condition=TextMentionTermination("TERMINATE"),
    )

    print("任务：写一篇关于 AutoGen 框架核心特性的技术简介\n")
    print("控制权流转：Planner（初始）→ Researcher → Writer → Planner → 结束\n")
    print("=" * 60)

    result = await team.run(
        task="写一篇关于 AutoGen 框架核心特性的技术简介，需要包含：是什么、核心架构、适用场景。"
    )

    print("\n" + "=" * 60)
    print("对话记录（注意控制权移交的时刻）：")
    print("=" * 60)
    for i, msg in enumerate(result.messages, 1):
        # 标记 HandoffMessage
        if isinstance(msg, HandoffMessage):
            print(f"\n[{i}] *** 控制权移交 ***")
            print(f"    {msg.source} → {msg.target}")
            print(f"    消息: {msg.content}")
        else:
            print(f"\n[{i}] {msg.source}（当前控制权持有者）")
            print("-" * 40)
            print(msg.content[:300] + "..." if len(msg.content) > 300 else msg.content)

    print("\n" + "=" * 60)
    print(f"总消息数: {len(result.messages)}")
    print("关键：每次 HandoffMessage 就是控制权转移的时刻")
    print("任意时刻只有一个 Agent 在工作，其他人完全沉默")


if __name__ == "__main__":
    asyncio.run(main())
