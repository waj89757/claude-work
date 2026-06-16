# AutoGen Demo - 三种多 Agent 模式

本项目用可运行代码展示 AutoGen v0.4 的三种核心协作模式。

## 模式说明

| 文件 | 模式 | 核心问题 |
|------|------|--------|
| `01_round_robin.py` | RoundRobin 轮流 | 谁来说话？按顺序轮流，无脑转圈 |
| `02_selector.py` | Selector 动态选人 | 谁来说话？外部 LLM 裁判动态决定 |
| `03_swarm.py` | Swarm 控制权移交 | 谁来说话？当前 Agent 自己决定移交给谁 |

## 快速开始

### 1. 安装依赖

```bash
cd autogen-demo
pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env 填入你的 API Key
```

也可以直接复用 `../advanced-rag/.env` 里的配置，运行脚本时会自动查找。

### 3. 运行 Demo

```bash
# Demo 1: RoundRobin 轮流对话
python 01_round_robin.py

# Demo 2: Selector 动态选人
python 02_selector.py

# Demo 3: Swarm 控制权移交
python 03_swarm.py
```

## 三种模式对比

```
RoundRobin:  A → B → C → A → B → C → ...  (固定顺序)
Selector:    A → ? → B → ? → A → ...       (每轮 LLM 裁判选人)
Swarm:       A → A 说"交给B" → B → B 说"交给C" → C → ...  (Agent 自己移交)
```

## 观察重点

运行时注意看输出里每条消息前的 `[Agent名字]`，理解消息流转顺序。
