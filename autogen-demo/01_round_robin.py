"""
Demo 1: RoundRobin GroupChat - 轮流发言模式

场景：产品经理、工程师、架构师三人讨论一个技术方案
特点：严格按顺序轮流，没有任何智能选人逻辑
     ProductManager → Engineer → Reviewer → ProductManager → ...

关键观察：
  - 不管上下文如何，到点就必须发言
  - 即使某个 Agent 没什么可说的，也会凑话
  - 适合角色对等、需要每人充分表达的场景
"""

import asyncio
import os
import sys
from dotenv import load_dotenv

# 优先加载本目录 .env，否则用 advanced-rag/.env
if os.path.exists(".env"):
    load_dotenv(".env")
elif os.path.exists("../advanced-rag/.env"):
    load_dotenv("../advanced-rag/.env")
    print("使用 advanced-rag/.env 配置\n")

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import MaxMessageTermination
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

    # 三个角色的 Agent
    product_manager = AssistantAgent(
        name="ProductManager",
        model_client=model_client,
        system_message=(
            "你是产品经理，负责提出需求和验收标准。"
            "每次发言简短（100字以内），重点说清楚业务需求和预期效果。"
            "不要重复别人说过的内容。"
        ),
    )

    engineer = AssistantAgent(
        name="Engineer",
        model_client=model_client,
        system_message=(
            "你是后端工程师，负责给出技术实现方案。"
            "每次发言简短（100字以内），用伪码或要点描述实现思路。"
            "不要重复别人说过的内容。"
        ),
    )

    reviewer = AssistantAgent(
        name="Reviewer",
        model_client=model_client,
        system_message=(
            "你是资深架构师，负责 review 方案的风险和可扩展性。"
            "每次发言简短（100字以内），指出潜在问题或给出改进建议。"
            "不要重复别人说过的内容。"
        ),
    )

    # RoundRobin: participants 列表顺序 = 发言顺序，转 3 圈（9条消息）后停止
    team = RoundRobinGroupChat(
        participants=[product_manager, engineer, reviewer],
        termination_condition=MaxMessageTermination(9),
    )

    print("任务：设计一个短视频平台的用户积分系统（日活100万）\n")
    print("发言顺序：ProductManager → Engineer → Reviewer → 循环\n")
    print("=" * 60)

    result = await team.run(
        task="设计一个短视频平台的用户积分系统，支持积分获取、消费、过期，日活100万。"
    )

    print("\n" + "=" * 60)
    print("对话记录：")
    print("=" * 60)
    for i, msg in enumerate(result.messages, 1):
        print(f"\n[{i}] {msg.source}")
        print("-" * 40)
        print(msg.content)

    print("\n" + "=" * 60)
    print(f"总消息数: {len(result.messages)}")
    print("注意：每个 Agent 严格轮流，不考虑上下文是否需要它发言")


if __name__ == "__main__":
    asyncio.run(main())
