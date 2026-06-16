# claude-work

个人 AI 工程实践学习仓库，系统性地探索 LLM 应用开发的核心技术栈。

涵盖从 RAG 到 Agent，从单框架到多框架对比，从理论理解到工程实践的完整学习路径。

---

## 项目结构

```
claude-work/
├── advanced-rag/        # 工业级 RAG 实现
├── autogen-demo/        # AutoGen 多 Agent 协作模式
├── react_agent/         # ReAct Agent 原理与框架实现
├── dspy_quickstart/     # DSPy 自动 Prompt 优化
├── lc_vs_lg/            # LangChain vs LangGraph 对比
└── mem0_demo/           # Mem0 长期记忆系统
```

---

## 模块详解

### 1. `advanced-rag/` — 工业级 RAG 实践

**核心问题**：Naive RAG 检索质量差，如何做到生产可用的 RAG？

实现了完整的 Advanced RAG Pipeline：

```
Query → Query 改写(HyDE) → 混合检索(BM25+Dense) → Reranker 重排序 → LLM 生成
```

| 模块 | 文件 | 作用 |
|------|------|------|
| 文档加载 | `src/document_loader.py` | 多格式文档解析，Sentence-window 分块策略 |
| 向量化 | `src/embeddings.py` | Embedding 模型封装（支持 OpenAI / 本地模型） |
| 检索器 | `src/retriever.py` | BM25 + Dense 混合检索，RRF 融合 |
| 重排序 | `src/reranker.py` | Cross-encoder 精细重排，过滤低相关文档 |
| Query 改写 | `src/query_transform.py` | HyDE（假设文档 Embedding）、多路扩展 |
| Pipeline | `src/rag_pipeline.py` | 组装完整 RAG 链路 |
| 配置 | `src/config.py` | 统一配置管理 |

**关键技术**：
- **Sentence-window Retrieval**：小块检索 + 大块返回，兼顾精准和完整
- **HyDE**：先让 LLM 生成假设答案，再用答案 Embedding 检索，解决 Query 与文档语义不对齐问题
- **Hybrid Search**：`final_score = α * BM25 + (1-α) * Dense`，专有名词和语义同时覆盖
- **Cross-encoder Rerank**：Bi-encoder 初筛 → Cross-encoder 精排，精度大幅提升

---

### 2. `autogen-demo/` — AutoGen 多 Agent 协作模式

**核心问题**：多个 Agent 如何协调工作？控制权如何流转？

展示 AutoGen v0.4 三种核心协作模式，每种模式对应不同的"谁来说话"决策机制：

| 文件 | 模式 | 控制权机制 |
|------|------|---------|
| `01_round_robin.py` | RoundRobin | 固定顺序轮流，简单无脑 |
| `02_selector.py` | Selector | 外部 LLM 裁判每轮动态决定谁发言 |
| `03_swarm.py` | Swarm | 当前 Agent 自己决定把控制权交给谁 |

```
RoundRobin:  A → B → C → A → B → C ...   (顺序固定)
Selector:    A → [LLM 裁判] → B → [LLM 裁判] → C   (动态选择)
Swarm:       A → A.handoff(B) → B → B.handoff(C) → C   (自主移交)
```

---

### 3. `react_agent/` — ReAct Agent 原理与实现

**核心问题**：Agent 是怎么"思考并行动"的？框架封装了什么？

用两种方式实现同一个 ReAct Agent，对比理解：

| 文件 | 实现方式 | 学习重点 |
|------|---------|---------|
| `react_manual.py` | 手写 ReAct 循环，零框架依赖 | 理解 Thought→Action→Observation 的完整原理 |
| `react_langchain.py` | LangChain `create_react_agent` | 理解框架对原理做了什么封装 |
| `deep_agent_demo.py` | LangChain DeepAgents | 主从式 Multi-Agent + Plan 规划 + Human-in-the-loop |

**ReAct 核心循环**：
```
Thought:      我知道什么？还缺什么信息？
Action:       调用工具
Observation:  工具返回真实结果（非 LLM 编造！）
Thought:      根据结果再次推理...
Final Answer: 信息足够时输出最终答案
```

**DeepAgents 亮点**：
- Parallel Tool Calling（一次 LLM 调用发起多个工具并行执行）
- 子 Agent 委托（主 Agent 通过 `task` 工具把子任务委托给专门的子 Agent）
- Middleware 拦截（`wrap_model_call` / `wrap_tool_call` 透视每一步 LLM 输入输出）
- Human-in-the-loop（`interrupt_on` + `checkpointer` 实现工具执行前人工审批）

---

### 4. `dspy_quickstart/` — DSPy 自动 Prompt 优化

**核心问题**：能不能不手写 Prompt，让框架自动找到最优 Prompt？

DSPy 的核心思路：把 Prompt 工程变成一个**优化问题**。

```python
# 声明任务（不写 Prompt）
class QA(dspy.Signature):
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()

# 给训练集 + 评价指标
optimizer = dspy.MIPROv2(metric=my_metric)
optimized = optimizer.compile(module, trainset=trainset)
# DSPy 自动找到最佳 Prompt + few-shot examples
```

| 文件 | 内容 |
|------|------|
| `01_basics.py` | Signature、Module、Predict、ChainOfThought 基础概念 |
| `02_optimizer.py` | Optimizer 核心：自动优化 Prompt |
| `03_see_prompt.py` | 查看 DSPy 实际发送的 Prompt |
| `04_optimize.py` | 完整优化流程 |
| `05_optimize_prompt.py` | Prompt 优化对比实验 |
| `06_rag.py` | DSPy 实现 RAG Pipeline |

---

### 5. `lc_vs_lg/` — LangChain vs LangGraph 对比

**核心问题**：LangChain 和 LangGraph 什么时候用哪个？

| 文件 | 内容 |
|------|------|
| `01_langchain_chain.py` | LangChain 传统 Chain：线性流程，适合简单任务 |
| `02_langgraph_agent.py` | LangGraph Agent：图结构，适合循环决策 |
| `03_with_langsmith.py` | LangSmith 可观测性：追踪 LLM 调用链路 |
| `04_multi_agent.py` | LangGraph 多 Agent：复杂协作场景 |

**核心区别**：
- LangChain：`A → B → C`（固定流程，像函数调用链）
- LangGraph：`节点 + 边 + 条件路由`（状态机，支持循环和分支）

---

### 6. `mem0_demo/` — Mem0 长期记忆

**核心问题**：LLM 对话默认无记忆，如何实现跨会话的长期记忆？

`demo.py` 演示 Mem0 的核心能力：自动提取对话中的关键信息并持久化，下次对话时自动召回相关记忆。

---

## 技术栈

| 类别 | 技术 |
|------|------|
| LLM 框架 | LangChain, LangGraph, AutoGen v0.4, DSPy, DeepAgents |
| 向量数据库 | ChromaDB |
| Embedding | OpenAI text-embedding-3, 本地模型 |
| 重排序 | Cross-encoder (sentence-transformers) |
| 记忆 | Mem0 |
| 语言 | Python 3.11+ |
| 模型 | OpenAI GPT-4o, GPT-4o-mini（兼容 API 格式） |

---

## 环境配置

各子项目均有独立的 `requirements.txt`，共用 `advanced-rag/.env` 的 API Key 配置：

```bash
# 以 advanced-rag 为例
cd advanced-rag
pip install -r requirements.txt
cp .env.example .env
# 编辑 .env，填入 OPENAI_API_KEY 和 OPENAI_BASE_URL
```

其他子项目（autogen-demo、react_agent 等）会自动查找 `../advanced-rag/.env`。

---

## 学习路径建议

```
1. advanced-rag        → 理解 RAG 工程化
2. react_agent         → 理解单 Agent 推理循环
3. autogen-demo        → 理解多 Agent 协作
4. lc_vs_lg            → 理解框架选型
5. dspy_quickstart     → 理解 Prompt 自动优化
6. mem0_demo           → 理解 Agent 记忆
```
