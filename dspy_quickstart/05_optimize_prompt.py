"""
DSPy COPRO 优化器：自动改写 Signature 的 instruction（System message 里的 objective）

BootstrapFewShot  → 只优化 few-shot demos（User message 里的示例）
COPRO            → 只优化 instruction（System message 里的 "your objective is: ..."）

运行流程：
1. 打印优化前的 prompt（原始 docstring）
2. COPRO 用 LLM 生成多个候选 instruction，在 trainset 上评分，选最好的
3. 打印优化后的 prompt（instruction 被替换）
"""

import dspy
from dspy.teleprompt import COPRO
from config import configure_dspy

lm = configure_dspy()

# ── 1. Signature ───────────────────────────────────────────────────
class SentimentClassifier(dspy.Signature):
    """Classify the sentiment of the given text."""   # ← COPRO 会改写这行
    text: str = dspy.InputField()
    sentiment: str = dspy.OutputField(desc="must be exactly 'positive' or 'negative'")

# ── 2. Program ─────────────────────────────────────────────────────
class SentimentProgram(dspy.Module):
    def __init__(self):
        self.classify = dspy.ChainOfThought(SentimentClassifier)

    def forward(self, text):
        return self.classify(text=text)

# ── 3. 训练数据 ────────────────────────────────────────────────────
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

# ── 4. metric ──────────────────────────────────────────────────────
def sentiment_metric(example, prediction, trace=None):
    return example.sentiment.lower() == prediction.sentiment.lower()

# ── 5. 打印优化前 ──────────────────────────────────────────────────
print("=" * 60)
print("【优化前】原始 instruction")
print("=" * 60)

program = SentimentProgram()
program(text="I really enjoyed this!")
print(lm.inspect_history(n=1))

# ── 6. COPRO 优化 ──────────────────────────────────────────────────
print("=" * 60)
print("【优化中】COPRO 正在生成候选 instruction 并评分 ...")
print("=" * 60)
print()

# breadth=3：每轮生成 3 个候选 instruction
# depth=2：迭代 2 轮（每轮基于上一轮最好的 instruction 继续改写）
optimizer = COPRO(
    metric=sentiment_metric,
    breadth=3,
    depth=2,
    verbose=True,  # 打印每个候选 instruction 和得分
)

optimized_program = optimizer.compile(
    SentimentProgram(),
    trainset=trainset,
    eval_kwargs={"num_threads": 1},
)

# ── 7. 打印优化后 ──────────────────────────────────────────────────
print()
print("=" * 60)
print("【优化后】新的 instruction（System message 变了）")
print("=" * 60)

optimized_program(text="I really enjoyed this!")
print(lm.inspect_history(n=1))

# ── 8. 直接看改写后的 instruction 文本 ────────────────────────────
print()
print("=" * 60)
print("【对比】instruction 前后变化")
print("=" * 60)
print(f"\n原始: Classify the sentiment of the given text.")



"""
优化前：
Classify the sentiment of the given text.
优化后：
You are given a piece of text. Determine the overall sentiment expressed by the author 
toward the main subject (not isolated words). Classify it into exactly one label from: 
positive, negative, or neutral.
Guidelines:
- Focus on the dominant attitude in the full text; if mixed, choose the prevailing sentiment.
- If the text is factual/objective with no clear opinion, choose neutral.
- If sarcasm/irony is present, infer the intended sentiment.
- Output only the single label (no explanations, no extra words, no punctuation).
"""


# 取优化后 program 里 Predict 的 signature instructions
optimized_sig = optimized_program.classify.predict.signature
print(f"优化后: {optimized_sig.instructions}")
