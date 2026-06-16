# DSPy 快速入门

## 安装

```bash
cd dspy_quickstart
pip install -r requirements.txt
export OPENAI_API_KEY=sk-xxx
```

## 运行示例

```bash
# 基础概念：Signature、Module、ChainOfThought
python 01_basics.py

# 核心能力：Optimizer 自动优化 prompt
python 02_optimizer.py
```

## 核心概念速查

| 概念 | 作用 | 类比 |
|-----|------|------|
| **Signature** | 声明输入输出 | 函数签名 |
| **Module** | 可组合的执行单元 | PyTorch nn.Module |
| **Predict** | 直接调用 LM | 前向传播 |
| **ChainOfThought** | 先推理再回答 | 带中间步骤 |
| **Optimizer** | 自动优化 prompt | 训练器 |

## DSPy 的价值

**传统方式**：
```
人工写 prompt → 测试 → 调整 → 测试 → 调整...
```

**DSPy 方式**：
```python
# 1. 声明任务
class QA(dspy.Signature):
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()

# 2. 给几个例子
trainset = [Example(...), Example(...)]

# 3. 自动优化
optimizer = dspy.MIPROv2(metric=my_metric)
optimized = optimizer.compile(module, trainset=trainset)
```

**结果**：DSPy 自动找到最佳 prompt + few-shot examples
