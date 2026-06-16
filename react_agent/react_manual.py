"""
ReAct Agent 手写实现 - 完全透明版

ReAct = Reasoning + Acting
核心循环：Thought → Action → Observation → Thought → Action → ... → Final Answer

这个文件手写了完整的 ReAct 循环，没有用任何框架封装，
目的是让每一步的 Thought/Action/Observation 都清晰可见。

场景：公司投资价值分析 Agent
Agent 需要自主决定：
  1. 先查什么信息
  2. 根据信息再查什么
  3. 何时信息足够可以给出最终答案

运行方式：
  python react_manual.py
"""

import os
import json
import re
from dotenv import load_dotenv
from openai import OpenAI

# 加载配置
if os.path.exists(".env"):
    load_dotenv(".env")
elif os.path.exists("../advanced-rag/.env"):
    load_dotenv("../advanced-rag/.env")

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
)
MODEL = os.getenv("LLM_MODEL", "gpt-4o")


# ============================================================
# 工具定义（模拟真实 API，返回假数据用于演示）
# ============================================================

def get_company_info(company_name: str) -> dict:
    """获取公司基本信息"""
    # 模拟数据库查询
    mock_data = {
        "比亚迪": {
            "full_name": "比亚迪股份有限公司",
            "industry": "新能源汽车、电池",
            "founded": "1995年",
            "employees": "约70万人",
            "main_products": "新能源汽车、动力电池、储能系统",
            "market_position": "全球新能源汽车销量第一（2023年）",
        },
        "宁德时代": {
            "full_name": "宁德时代新能源科技股份有限公司",
            "industry": "动力电池",
            "founded": "2011年",
            "employees": "约10万人",
            "main_products": "动力电池、储能电池",
            "market_position": "全球动力电池市占率第一",
        },
    }
    data = mock_data.get(company_name, {"error": f"未找到 {company_name} 的信息"})
    return data


def get_financial_data(company_name: str, year: str = "2023") -> dict:
    """获取公司财务数据"""
    mock_data = {
        "比亚迪": {
            "revenue": "6023亿元",
            "revenue_growth": "+42%",
            "net_profit": "300亿元",
            "profit_growth": "+81%",
            "gross_margin": "22.1%",
            "pe_ratio": "约25倍",
            "debt_ratio": "65%",
            "cash_flow": "正向，经营现金流约400亿",
        },
        "宁德时代": {
            "revenue": "4009亿元",
            "revenue_growth": "+22%",
            "net_profit": "441亿元",
            "profit_growth": "+44%",
            "gross_margin": "22.9%",
            "pe_ratio": "约20倍",
            "debt_ratio": "55%",
            "cash_flow": "正向，经营现金流约350亿",
        },
    }
    data = mock_data.get(company_name, {"error": f"未找到 {company_name} 的财务数据"})
    data["year"] = year
    return data


def get_news_sentiment(company_name: str) -> dict:
    """获取近期新闻舆情"""
    mock_data = {
        "比亚迪": {
            "sentiment": "偏正面",
            "score": 0.72,
            "recent_news": [
                "比亚迪1月销量再创新高，同比增长50%",
                "比亚迪宣布进入欧洲市场，建立本地化工厂",
                "比亚迪与多家车企谈判电池供应合作",
            ],
            "risks": [
                "欧盟对中国电动车加征关税",
                "国内市场价格战激烈",
            ],
        },
        "宁德时代": {
            "sentiment": "中性偏正面",
            "score": 0.61,
            "recent_news": [
                "宁德时代固态电池研发取得突破",
                "海外建厂计划推进，匈牙利工厂投产",
                "与多家跨国车企续签供货协议",
            ],
            "risks": [
                "比亚迪等车企自研电池，客户集中度风险",
                "原材料碳酸锂价格波动",
            ],
        },
    }
    data = mock_data.get(company_name, {"sentiment": "无数据", "score": 0.5, "recent_news": [], "risks": []})
    return data


def get_competitor_analysis(company_name: str) -> dict:
    """获取竞争对手分析"""
    mock_data = {
        "比亚迪": {
            "main_competitors": ["特斯拉", "吉利", "上汽", "理想", "蔚来"],
            "competitive_advantages": [
                "垂直整合：自研电池、芯片、电机",
                "成本控制能力强",
                "产品线覆盖全价位段",
            ],
            "weaknesses": [
                "品牌高端化仍在推进中",
                "海外市场刚起步",
            ],
        },
        "宁德时代": {
            "main_competitors": ["比亚迪电池", "LG新能源", "松下", "中航锂电"],
            "competitive_advantages": [
                "技术领先，多项专利",
                "与主流车企绑定深",
                "规模效应显著",
            ],
            "weaknesses": [
                "客户垂直整合趋势带来依赖风险",
                "产能快速扩张带来折旧压力",
            ],
        },
    }
    return mock_data.get(company_name, {"error": f"未找到竞争分析数据"})


# ============================================================
# 工具注册表（Agent 能调用的所有工具）
# ============================================================

TOOLS = {
    "get_company_info": {
        "func": get_company_info,
        "description": "获取公司基本信息，包括行业、规模、主营产品等",
        "params": {"company_name": "公司名称（中文）"},
    },
    "get_financial_data": {
        "func": get_financial_data,
        "description": "获取公司财务数据，包括营收、利润、增长率、PE等",
        "params": {"company_name": "公司名称（中文）", "year": "年份（默认2023）"},
    },
    "get_news_sentiment": {
        "func": get_news_sentiment,
        "description": "获取公司近期新闻和市场舆情，包括正面消息和风险点",
        "params": {"company_name": "公司名称（中文）"},
    },
    "get_competitor_analysis": {
        "func": get_competitor_analysis,
        "description": "获取公司竞争态势分析，包括竞争对手、优势和弱点",
        "params": {"company_name": "公司名称（中文）"},
    },
}

# ============================================================
# ReAct Prompt 构建
# ============================================================

REACT_SYSTEM_PROMPT = """你是一个专业的股票研究分析师 Agent。

你可以使用以下工具来收集信息：

{tool_descriptions}

你必须严格按照以下格式进行推理和行动，每次只执行一个步骤：

Thought: [你的思考过程，分析当前已有的信息，决定下一步做什么]
Action: [工具名称]
Action Input: {{"参数名": "参数值"}}

当你收集了足够的信息，可以给出最终答案时，使用：
Thought: [最终总结性思考]
Final Answer: [你的完整分析和投资建议]

重要规则：
1. 每次只输出一个 Thought + Action 或者 Thought + Final Answer
2. 不要一次输出多个 Action
3. Action 必须是工具列表中存在的工具名
4. Action Input 必须是合法的 JSON 格式
5. 在给出 Final Answer 之前，至少要调用3个不同的工具
"""


def build_tool_descriptions() -> str:
    desc = ""
    for name, info in TOOLS.items():
        params = ", ".join([f"{k}（{v}）" for k, v in info["params"].items()])
        desc += f"- {name}({params}): {info['description']}\n"
    return desc


# ============================================================
# ReAct 核心执行循环
# ============================================================

def execute_action(action_name: str, action_input: dict) -> str:
    """执行工具调用，返回 Observation"""
    if action_name not in TOOLS:
        return f"错误：工具 '{action_name}' 不存在，可用工具：{list(TOOLS.keys())}"
    
    tool = TOOLS[action_name]
    try:
        result = tool["func"](**action_input)
        return json.dumps(result, ensure_ascii=False, indent=2)
    except Exception as e:
        return f"工具执行出错：{str(e)}"


def parse_llm_output(text: str) -> dict:
    """
    解析 LLM 输出，提取 Thought/Action/ActionInput/FinalAnswer
    返回格式：{"thought": ..., "action": ..., "action_input": ..., "final_answer": ...}
    """
    result = {"thought": "", "action": None, "action_input": None, "final_answer": None}

    # 提取 Thought
    thought_match = re.search(r"Thought:\s*(.+?)(?=Action:|Final Answer:|$)", text, re.DOTALL)
    if thought_match:
        result["thought"] = thought_match.group(1).strip()

    # 提取 Final Answer
    final_match = re.search(r"Final Answer:\s*(.+)", text, re.DOTALL)
    if final_match:
        result["final_answer"] = final_match.group(1).strip()
        return result

    # 提取 Action
    action_match = re.search(r"Action:\s*(\w+)", text)
    if action_match:
        result["action"] = action_match.group(1).strip()

    # 提取 Action Input（JSON）
    input_match = re.search(r"Action Input:\s*(\{.+?\})", text, re.DOTALL)
    if input_match:
        try:
            result["action_input"] = json.loads(input_match.group(1))
        except json.JSONDecodeError:
            # 尝试修复简单格式问题
            raw = input_match.group(1)
            result["action_input"] = {"company_name": re.search(r'"([^"]+)"', raw).group(1) if re.search(r'"([^"]+)"', raw) else ""}

    return result


def run_react_agent(task: str, max_steps: int = 10) -> str:
    """
    执行 ReAct 循环
    
    每一轮：
    1. 把当前 scratchpad（所有历史 Thought/Action/Observation）发给 LLM
    2. LLM 输出下一步的 Thought + Action（或 Final Answer）
    3. 执行 Action，得到 Observation
    4. 把 Observation 追加到 scratchpad
    5. 重复直到 Final Answer 或达到最大步数
    """
    
    system_prompt = REACT_SYSTEM_PROMPT.format(
        tool_descriptions=build_tool_descriptions()
    )
    
    # scratchpad 记录所有的 Thought/Action/Observation 历史
    scratchpad = ""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"任务：{task}"},
    ]
    
    print("\n" + "=" * 70)
    print(f"📋 任务: {task}")
    print("=" * 70)
    
    for step in range(1, max_steps + 1):
        print(f"\n{'─' * 70}")
        print(f"⚙️  步骤 {step}")
        print(f"{'─' * 70}")
        
        # 如果已有 scratchpad，拼接到最后一个 user 消息里
        if scratchpad:
            # 把历史 scratchpad 追加到对话
            messages_with_scratch = messages[:-1] + [
                {"role": "user", "content": messages[-1]["content"] + "\n\n" + scratchpad.strip()}
            ]
        else:
            messages_with_scratch = messages
        
        # 调用 LLM
        # 注意：如果模型是 reasoning 模型（如 Qwen3 思考模式），需要更大的 max_tokens
        # 因为思考过程会消耗大量 token，需要留足够空间给实际输出
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages_with_scratch,
            max_tokens=2000,
            temperature=0,
            stop=["Observation:"],  # 关键：遇到 Observation: 就停止，让我们填入真实的工具结果
        )
        
        llm_output = response.choices[0].message.content.strip()
        
        # 解析 LLM 输出
        parsed = parse_llm_output(llm_output)
        
        # 打印 Thought
        if parsed["thought"]:
            print(f"\n💭 Thought:")
            print(f"   {parsed['thought']}")
        
        # 如果是 Final Answer，结束循环
        if parsed["final_answer"]:
            print(f"\n✅ Final Answer:")
            print(f"{'─' * 70}")
            print(parsed["final_answer"])
            print(f"{'─' * 70}")
            return parsed["final_answer"]
        
        # 如果有 Action，执行工具
        if parsed["action"] and parsed["action_input"] is not None:
            print(f"\n🔧 Action: {parsed['action']}")
            print(f"   Input: {json.dumps(parsed['action_input'], ensure_ascii=False)}")
            
            # 执行工具
            observation = execute_action(parsed["action"], parsed["action_input"])
            
            print(f"\n👁️  Observation:")
            # 格式化打印观察结果
            for line in observation.split('\n'):
                print(f"   {line}")
            
            # 把这一步追加到 scratchpad
            scratchpad += f"\nThought: {parsed['thought']}\nAction: {parsed['action']}\nAction Input: {json.dumps(parsed['action_input'], ensure_ascii=False)}\nObservation: {observation}\n"
            
        else:
            # LLM 输出格式不对，给一个提示
            print(f"\n⚠️  LLM 输出格式异常，原始输出：")
            print(llm_output)
            scratchpad += f"\n{llm_output}\n"
    
    return "达到最大步数，未能得出最终答案"


# ============================================================
# 主程序
# ============================================================

if __name__ == "__main__":
    print("=" * 70)
    print("ReAct Agent 手写实现演示")
    print("核心思想：Thought → Action → Observation → Thought → ...")
    print("=" * 70)
    
    # 任务 1：单公司分析
    result = run_react_agent(
        task="请分析比亚迪公司是否值得长期投资，需要从基本面、财务状况、舆情风险、竞争格局四个维度分析"
    )
    
    print("\n\n" + "=" * 70)
    print("说明：")
    print("- 💭 Thought：Agent 的推理过程，决定下一步做什么")
    print("- 🔧 Action：调用哪个工具，传什么参数")
    print("- 👁️  Observation：工具返回的真实结果")
    print("- ✅ Final Answer：收集足够信息后的最终结论")
    print("- 注意观察 Agent 如何根据每次 Observation 调整下一步策略")
    print("=" * 70)
