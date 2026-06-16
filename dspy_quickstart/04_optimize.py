x"""
DSPy 自动优化示例 + 白盒打印每次 Prompt

任务：情感分类（positive / negative）
优化器：BootstrapFewShot —— 自动找好的 few-shot 示例注入到 prompt 里

运行流程：
1. 先打印"优化前"的 prompt（无 few-shot，只有 instruction）
2. 在 trainset 上跑 BootstrapFewShot
3. 打印"优化后"的 prompt（自动注入了高质量示例）
"""

import dspy
from dspy.teleprompt import BootstrapFewShot
from config import configure_dspy

lm = configure_dspy()

# ── 1. 定义 Signature ──────────────────────────────────────────────
class SentimentClassifier(dspy.Signature):
    """Classify the sentiment of the given text."""
    text: str = dspy.InputField()
    sentiment: str = dspy.OutputField(desc="must be exactly 'positive' or 'negative'")

# ── 2. 定义 Program（用 ChainOfThought）────────────────────────────
class SentimentProgram(dspy.Module):
    def __init__(self):
        self.classify = dspy.ChainOfThought(SentimentClassifier)

    def forward(self, text):
        return self.classify(text=text)

# ── 3. 训练数据（10 条带标签的样本）──────────────────────────────────
trainset = [
    dspy.Example(text="I love this product! It's amazing.", sentiment="positive").with_inputs("text"),
    dspy.Example(text="This is terrible. Worst purchase ever.", sentiment="negative").with_inputs("text"),
    dspy.Example(text="Absolutely fantastic experience!", sentiment="positive").with_inputs("text"),
    dspy.Example(text="Broken on arrival. Very disappointed.", sentiment="negative").with_inputs("text"),
    dspy.Example(text="Great quality and fast shipping.", sentiment="positive").with_inputs("text"),
    dspy.Example(text="Waste of money. Don't buy this.", sentiment="negative").with_inputs("text"),
    dspy.Example(text="Exceeded my expectations. Highly recommend!", sentiment="positive").with_inputs("text"),
    dspy.Example(text="Poor customer service. Never again.", sentiment="negative").with_inputs("text"),
]

# ── 4. 评估函数（metric）──────────────────────────────────────────
def sentiment_metric(example, prediction, trace=None):
    """返回 True/False：预测是否正确"""
    return example.sentiment.lower() == prediction.sentiment.lower()

# ── 5. 打印"优化前"的 Prompt ─────────────────────────────────────
print("=" * 60)
print("【优化前】执行一次，看原始 Prompt")
print("=" * 60)

program = SentimentProgram()
result = program(text="This movie was absolutely brilliant!")
print(lm.inspect_history(n=1))
print(f"\n>>> 预测结果: {result.sentiment}\n")

# ── 6. 运行 BootstrapFewShot 优化 ────────────────────────────────
print("=" * 60)
print("【优化中】BootstrapFewShot 开始跑 trainset ...")
print("(每条样本都会调用 LLM，可以看到多次 prompt)")
print("=" * 60)
print()

# max_bootstrapped_demos=3 表示最多注入 3 条 few-shot 示例
optimizer = BootstrapFewShot(
    metric=sentiment_metric,
    max_bootstrapped_demos=3,
    max_labeled_demos=3,
)

optimized_program = optimizer.compile(
    student=SentimentProgram(),
    trainset=trainset,
)

# ── 7. 打印"优化后"的 Prompt ─────────────────────────────────────
print()
print("=" * 60)
print("【优化后】执行一次，看注入了 few-shot 后的 Prompt")
print("=" * 60)

result2 = optimized_program(text="This movie was absolutely brilliant!")
print(lm.inspect_history(n=1))
print(f"\n>>> 预测结果: {result2.sentiment}\n")

# ── 8. 对比：查看优化后注入了哪些 demos ──────────────────────────
print("=" * 60)
print("【注入的 Few-shot Demos】（优化器自动选出的）")
print("=" * 60)
demos = optimized_program.classify.demos if hasattr(optimized_program.classify, 'demos') else optimized_program.classify.predict.demos
if demos:
    for i, demo in enumerate(demos, 1):
        print(f"\nDemo {i}:")
        print(f"  text      : {demo.text}")
        print(f"  sentiment : {demo.sentiment}")
else:
    print("（没有注入 demos）")
