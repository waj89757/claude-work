"""
Demo 2: SelectorGroupChat - 动态选人模式（含决策过程可视化）

场景：同样是产品经理、工程师、架构师讨论技术方案
特点：每轮发言后，由一个外部 LLM 裁判决定"下一个谁来说"
     没有固定顺序，谁最适合说下一句，谁来说

关键观察：
  - 对比 Demo 1，发言顺序会根据内容变化
  - 日志里会打印 [🤔 裁判决策] 块，显示 LLM 裁判的完整推理过程
  - 代价：每轮多一次 LLM 调用（选人用的）
  - 适合角色职责有明确边界、需要"接话"的场景
"""

import asyncio
import os
from typing import Sequence
from dotenv import load_dotenv

if os.path.exists(".env"):
    load_dotenv(".env")
elif os.path.exists("../advanced-rag/.env"):
    load_dotenv("../advanced-rag/.env")
    print("使用 advanced-rag/.env 配置\n")

from autogen_agentchat.agents import AssistantAgent, BaseChatAgent
from autogen_agentchat.teams import SelectorGroupChat
from autogen_agentchat.conditions import MaxMessageTermination
from autogen_agentchat.messages import BaseChatMessage
from autogen_ext.models.openai import OpenAIChatCompletionClient
from openai import OpenAI

# 选人规则 prompt（用于自定义 selector_func）
SELECTOR_SYSTEM_PROMPT = """你是一个对话协调员。根据当前对话内容，选择最适合下一个发言的参与者。

规则：
- 如果上一条消息提出了业务需求，选 Engineer 来回应技术方案
- 如果上一条消息是技术方案，选 Reviewer 来 review
- 如果 Reviewer 指出了问题，选 Engineer 修改方案
- 如果技术方案已经 OK，选 ProductManager 确认是否满足需求
- 避免同一个人连续发言两次

只返回一个名字：ProductManager、Engineer 或 Reviewer，不要有其他内容。"""


def make_visible_selector(api_key: str, base_url: str, model: str):
    """
    返回一个自定义 selector_func。
    这个函数替代 SelectorGroupChat 内部的 LLM 选人逻辑，
    但在调用 LLM 前后打印完整的推理过程，让决策透明可见。
    """
    raw_client = OpenAI(api_key=api_key, base_url=base_url)

    def selector_func(messages: Sequence[BaseChatMessage], agents: Sequence[BaseChatAgent]) -> str | None:
        agent_names = [a.name for a in agents]

        # 把对话历史格式化成可读形式给裁判看
        history_text = ""
        for msg in messages[-6:]:  # 只看最近 6 条，避免 context 过长
            history_text += f"[{msg.source}]: {msg.content[:150]}\n"

        user_prompt = f"当前参与者：{agent_names}\n\n最近对话：\n{history_text}\n下一个发言者是谁？"

        print(f"\n{'─'*60}")
        print(f"🤔 [裁判决策] 正在分析对话，决定谁来发言...")
        print(f"   上一条发言者: {messages[-1].source if messages else '无'}")
        print(f"   可选参与者: {agent_names}")

        # 调用 LLM 做选人决策
        response = raw_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SELECTOR_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=20,
            temperature=0,
        )
        decision = response.choices[0].message.content.strip()

        # 匹配最接近的 agent 名字（容错处理）
        selected = None
        for name in agent_names:
            if name.lower() in decision.lower():
                selected = name
                break

        # 如果 LLM 输出了无法匹配的名字，fallback 到第一个
        if selected is None:
            selected = agent_names[0]
            print(f"   LLM 原始输出: '{decision}' (无法匹配，fallback 到 {selected})")
        else:
            print(f"   LLM 原始输出: '{decision}'")
            print(f"   ✅ 裁判选择: {selected}")
        print(f"{'─'*60}\n")

        return selected

    return selector_func


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

    product_manager = AssistantAgent(
        name="ProductManager",
        model_client=model_client,
        system_message=(
            "你是产品经理，负责：1) 明确业务需求 2) 确认验收标准 3) 对方案提出业务层面的质疑。"
            "每次发言简短（100字以内）。"
        ),
    )

    engineer = AssistantAgent(
        name="Engineer",
        model_client=model_client,
        system_message=(
            "你是后端工程师，负责：1) 给出技术方案 2) 评估可行性 3) 回应技术层面的质疑。"
            "每次发言简短（100字以内），用要点或伪码。"
        ),
    )

    reviewer = AssistantAgent(
        name="Reviewer",
        model_client=model_client,
        system_message=(
            "你是架构师，负责：1) review 技术方案 2) 指出扩展性问题 3) 给出最终技术决策。"
            "每次发言简短（100字以内）。"
        ),
    )

    # 用自定义 selector_func 替代内部 LLM 选人
    # 效果一样，但决策过程完全可见
    team = SelectorGroupChat(
        participants=[product_manager, engineer, reviewer],
        model_client=model_client,
        termination_condition=MaxMessageTermination(9),
        selector_func=make_visible_selector(api_key, base_url, model),
    )

    print("任务：设计一个短视频平台的用户积分系统（日活100万）\n")
    print("发言顺序：由 LLM 裁判根据上下文动态决定（决策过程可见）\n")
    print("=" * 60)

    result = await team.run(
        task="设计一个短视频平台的用户积分系统，支持积分获取、消费、过期，日活100万。"
    )

    print("\n" + "=" * 60)
    print("完整对话记录：")
    print("=" * 60)
    for i, msg in enumerate(result.messages, 1):
        print(f"\n[{i}] {msg.source}")
        print("-" * 40)
        print(msg.content)

    print("\n" + "=" * 60)
    print(f"总消息数: {len(result.messages)}")
    print("注意：每次 [🤔 裁判决策] 就是 LLM 做一次选人判断的时刻")


if __name__ == "__main__":
    asyncio.run(main())
