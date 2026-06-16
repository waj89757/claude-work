"""
白盒查看 DSPy 生成的 Prompt
"""

import dspy
from config import configure_dspy

lm = configure_dspy()

# 定义 Signature
class QA(dspy.Signature):
    """Answer a question."""
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()

print("=" * 60)
print("1. dspy.Predict 的 Prompt")
print("=" * 60)

predict = dspy.Predict(QA)
result = predict(question="What is 2+2?")
print(lm.inspect_history(n=1))

print("\n")
print("=" * 60)
print("2. dspy.ChainOfThought 的 Prompt")
print("=" * 60)

cot = dspy.ChainOfThought(QA)
result = cot(question="What is 2+2?")
print(lm.inspect_history(n=1))
