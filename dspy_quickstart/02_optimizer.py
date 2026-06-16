"""
DSPy Optimizer 示例
展示如何用少量标注数据自动优化 prompt

这是 DSPy 的核心价值：
- 传统方式：人工调 prompt → 费时费力
- DSPy 方式：给几个例子 → 自动优化出最佳 prompt
"""

import dspy
from config import configure_dspy

# ============================================================
# Step 1: 配置 LM（从 .env 加载）
# ============================================================
lm = configure_dspy()
print(f"✅ LM configured: {lm.model}")

# ============================================================
# Step 2: 定义任务
# ============================================================

class Sentiment(dspy.Signature):
    """Classify the sentiment of a review."""
    review: str = dspy.InputField()
    sentiment: str = dspy.OutputField(desc="positive or negative")

# 基础模块
classifier = dspy.Predict(Sentiment)

# ============================================================
# Step 3: 准备训练数据 (只需要几个例子!)
# ============================================================

trainset = [
    dspy.Example(
        review="This product is amazing! Best purchase ever.",
        sentiment="positive"
    ).with_inputs("review"),
    
    dspy.Example(
        review="Terrible quality, broke after one day.",
        sentiment="negative"
    ).with_inputs("review"),
    
    dspy.Example(
        review="Love it! Exceeded my expectations.",
        sentiment="positive"
    ).with_inputs("review"),
    
    dspy.Example(
        review="Waste of money, very disappointed.",
        sentiment="negative"
    ).with_inputs("review"),
]

# ============================================================
# Step 4: 定义评估指标
# ============================================================

def accuracy_metric(example, prediction, trace=None):
    return example.sentiment.lower() == prediction.sentiment.lower()

# ============================================================
# Step 5: 使用 Optimizer 自动优化
# ============================================================

# BootstrapFewShot: 自动选择最佳 few-shot 示例
optimizer = dspy.BootstrapFewShot(metric=accuracy_metric, max_bootstrapped_demos=2)

# 编译/优化
optimized_classifier = optimizer.compile(classifier, trainset=trainset)

# ============================================================
# Step 6: 测试优化后的模型
# ============================================================

test_reviews = [
    "Absolutely fantastic! Would buy again.",
    "Not worth the price, very poor quality.",
    "The delivery was fast and the product works great!",
]

print("Testing optimized classifier:\n")
for review in test_reviews:
    result = optimized_classifier(review=review)
    print(f"Review: {review[:50]}...")
    print(f"Sentiment: {result.sentiment}\n")

# ============================================================
# Step 7: 查看优化后的 prompt
# ============================================================

print("="*50)
print("Optimized prompt structure:")
print(lm.inspect_history(n=1))

# ============================================================
# Step 8: 保存优化后的程序
# ============================================================

optimized_classifier.save("optimized_sentiment.json")
print("\n✅ Saved to optimized_sentiment.json")
print("可以用 dspy.Predict(Sentiment).load('optimized_sentiment.json') 加载")
