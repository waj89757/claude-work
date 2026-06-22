# RLM Demo - 递归语言模型示例

## 什么是 RLM（Recursive Language Model）

RLM 是一种让 LLM **递归调用自身**来解决复杂问题的设计模式。

核心思想：当问题太复杂时，LLM 先把它拆解成子问题，每个子问题再次调用 LLM 解决，直到问题简单到可以直接回答，最后再合并所有子答案。

```
传统 LLM：问题 → LLM → 答案（一次调用）

RLM：
  问题
    ↓ LLM 拆解
  [子问题A, 子问题B, 子问题C]
    ↓ 每个子问题再次调用 LLM
  [子子问题A1, 子子问题A2] ... （继续递归）
    ↓ 达到 base case，直接回答
  基础答案
    ↓ 层层合并
  最终答案
```

## 与其他 Agent 模式的关系

| 模式 | 调用结构 | 适合场景 |
|------|---------|---------|
| 普通 LLM | 线性，一次调用 | 简单问答 |
| ReAct | 循环，Thought→Action→Observation | 需要工具的任务 |
| Plan & Execute | 先规划再串行执行 | 多步骤任务 |
| **RLM** | **树状，递归拆解合并** | **层级复杂问题、长文档** |

## 快速开始

```bash
cd rlm_demo
pip install openai python-dotenv
python rlm_demo.py
```

环境变量复用 `../advanced-rag/.env`。

## 代码结构

```
rlm_demo.py
├── llm_call()           # 单次 LLM 调用（叶子节点）
├── rlm_solve()          # 核心递归函数
│   ├── Base Case 1：超过 max_depth → 强制直接回答
│   ├── Base Case 2：LLM 判断可直接回答 → 返回答案
│   └── Recursive Case：拆解 → 递归 → 合并
├── demo_simple()        # 示例1：简单问题（不需要递归）
├── demo_complex()       # 示例2：复杂问题（自动递归）
└── demo_hierarchical_summary()  # 示例3：层级摘要
```

## RLM 的实际应用

1. **层级文档摘要**：先摘要段落 → 再摘要章节 → 再摘要全书（处理超长文档）
2. **复杂推理**：数学证明、多步逻辑推导
3. **代码生成**：先设计架构 → 生成模块 → 生成函数
4. **知识图谱构建**：大主题 → 子主题 → 具体概念
