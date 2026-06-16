# ReAct Agent 学习项目

用两种方式实现 ReAct Agent，帮你理解 `Thought → Action → Observation` 循环的本质。

## 文件说明

| 文件 | 方式 | 学习重点 |
|------|------|--------|
| `react_manual.py` | 手写 ReAct 循环，零框架 | 理解原理，每一步完全透明 |
| `react_langchain.py` | LangChain `create_react_agent` | 理解框架封装了什么 |

## 核心概念

```
ReAct = Reasoning（推理）+ Acting（行动）

循环：
  Thought:      Agent 思考：我现在知道什么？下一步需要什么？
  Action:       调用工具
  Action Input: 工具参数
  Observation:  工具返回的真实结果（不是 LLM 编的！）
  Thought:      根据 Observation 再次思考...
  ...
  Final Answer: 收集足够信息后给出最终答案
```

## 场景设计

**公司投资价值分析 Agent**

可以调用的工具：
- `get_company_info` — 公司基本信息
- `get_financial_data` — 财务数据
- `get_news_sentiment` — 新闻舆情
- `get_competitor_analysis` — 竞争格局
- `compare_companies` — 两公司对比（LangChain 版额外工具）

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 手写版（推荐先看这个，理解原理）
python react_manual.py

# LangChain 框架版（对比看框架封装了什么）
python react_langchain.py
```

## 观察重点

运行 `react_manual.py` 时，注意：
1. **💭 Thought**：Agent 每次的推理内容是否合理？
2. **🔧 Action**：Agent 选择的工具顺序是否有逻辑？
3. **👁️ Observation**：工具返回的真实数据（不是 LLM 幻想的）
4. **迭代**：Agent 如何根据前一步的 Observation 调整下一步策略

关键洞察：**Observation 是真实的工具执行结果，这正是 ReAct 比纯 CoT 更可靠的原因**

## 和 AutoGen 的关系

```
ReAct Agent（单个 Agent 内部的推理循环）
    ↑
AutoGen AssistantAgent（内部用 ReAct 或类似机制驱动）
    ↑
AutoGen GroupChat（多个 AssistantAgent 互相对话）
```

ReAct 是"一个 Agent 怎么思考"，AutoGen 是"多个 Agent 怎么协作"。
