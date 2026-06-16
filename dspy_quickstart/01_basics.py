"""
DSPy 快速入门示例
展示 DSPy 的核心概念：Signature、Module、Optimizer

配置说明：
- 修改 .env 文件中的 LLM_API_KEY、LLM_API_BASE、LLM_MODEL
"""

import dspy
from config import configure_dspy

# ============================================================
# Step 1: 配置 LM（从 .env 加载）
# ============================================================
lm = configure_dspy()
print(f"✅ LM configured: {lm.model}")

# ============================================================
# Step 2: 定义 Signature (输入 → 输出 的声明式规范)
# ============================================================

class QA(dspy.Signature):
    """Answer a question with a short factoid answer."""
    question: str = dspy.InputField()
    answer: str = dspy.OutputField(desc="1-2 sentences")


# ============================================================
# Step 3: 使用 Module 执行
# ============================================================

# dspy.Predict 是最基础的 Module，直接调用 LM
qa = dspy.Predict(QA)

# 调用
response = qa(question="What is the capital of France?")
print(f"Answer: {response.answer}")

# ============================================================
# Step 4: Chain of Thought (推理链)
# ============================================================

# ChainOfThought 会自动让 LM 先输出推理过程
cot_qa = dspy.ChainOfThought(QA)

response = cot_qa(question="If a train travels 120km in 2 hours, what is its speed?")
print(f"\nReasoning: {response.reasoning}")
print(f"Answer: {response.answer}")

# ============================================================
# Step 5: 自定义 Module (组合多个步骤)
# ============================================================

class RAGModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.retrieve = dspy.Predict("question -> context")
        self.generate = dspy.ChainOfThought("context, question -> answer")
    
    def forward(self, question):
        context = self.retrieve(question=question).context
        answer = self.generate(context=context, question=question).answer
        return dspy.Prediction(context=context, answer=answer)

# 使用
rag = RAGModule()
result = rag(question="What causes rainbows?")
print(f"\nRAG Context: {result.context}")
print(f"RAG Answer: {result.answer}")

print("\n" + "="*50)
print("✅ DSPy 基础示例完成!")
print("下一步：运行 02_optimizer.py 体验自动优化")
