"""
DSPy RAG 完整示例：检索 + 生成 + 自动优化

知识库：用一个假的本地检索器模拟（不需要搭建向量数据库）
优化器：BootstrapFewShot，自动优化生成答案的 few-shot demos

运行流程：
1. 定义本地知识库 + 自定义检索器
2. 定义 RAG Program
3. 优化前跑一次，打印 prompt
4. 用 BootstrapFewShot 优化
5. 优化后跑一次，打印 prompt（对比 few-shot 注入前后）
"""

import dspy
from dspy.teleprompt import BootstrapFewShot
from config import configure_dspy

lm = configure_dspy()

# ── 1. 本地知识库（模拟向量数据库）────────────────────────────────
KNOWLEDGE_BASE = [
    {"id": 0, "text": "DSPy is a framework for programming—not prompting—language models. It lets you compose LM calls into pipelines and automatically optimize prompts."},
    {"id": 1, "text": "DSPy Signatures define the input and output fields of a module. They are used to generate prompts automatically."},
    {"id": 2, "text": "DSPy optimizers (teleprompters) like BootstrapFewShot and MIPROv2 automatically improve prompts using training data."},
    {"id": 3, "text": "RAG stands for Retrieval-Augmented Generation. It retrieves relevant documents and uses them as context for the LLM to answer questions."},
    {"id": 4, "text": "ColBERT is a retrieval model that uses late interaction between query and document embeddings to rank passages."},
    {"id": 5, "text": "DSPy ChainOfThought adds a reasoning field before the answer, forcing the LLM to think step by step."},
    {"id": 6, "text": "Python is a high-level programming language known for its readability and simplicity. It is widely used in AI and data science."},
    {"id": 7, "text": "The metric function in DSPy takes an example and a prediction, returning a score that optimizers use to evaluate prompt quality."},
]

def simple_retrieve(query: str, k: int = 3) -> list[str]:
    """
    简单的关键词匹配检索（模拟向量检索）
    真实项目里替换成 Weaviate / Pinecone / ColBERT 等
    """
    query_words = set(query.lower().split())
    scored = []
    for doc in KNOWLEDGE_BASE:
        doc_words = set(doc["text"].lower().split())
        overlap = len(query_words & doc_words)
        scored.append((overlap, doc["text"]))
    scored.sort(reverse=True)
    return [text for _, text in scored[:k]]

# ── 2. 自定义 Retriever（接入 DSPy）──────────────────────────────
class LocalRetriever(dspy.Retrieve):
    """把本地检索函数包装成 DSPy Retrieve 模块"""
    def __init__(self, k=3):
        super().__init__(k=k)
        self.k = k

    def forward(self, query: str):
        passages = simple_retrieve(query, self.k)
        return dspy.Prediction(passages=passages)

# ── 3. Signature ───────────────────────────────────────────────────
class GenerateAnswer(dspy.Signature):
    """Answer questions based on the provided context."""
    context: list[str] = dspy.InputField(desc="relevant passages retrieved from knowledge base")
    question: str = dspy.InputField()
    answer: str = dspy.OutputField(desc="a concise answer based on the context")

# ── 4. RAG Program ─────────────────────────────────────────────────
class RAG(dspy.Module):
    def __init__(self, num_passages=3):
        super().__init__()
        self.retrieve = LocalRetriever(k=num_passages)
        self.generate_answer = dspy.ChainOfThought(GenerateAnswer)

    def forward(self, question):
        context = self.retrieve(question).passages
        prediction = self.generate_answer(context=context, question=question)
        return dspy.Prediction(context=context, answer=prediction.answer)

# ── 5. 训练数据（问题 + 标准答案）────────────────────────────────
trainset = [
    dspy.Example(
        question="What is DSPy?",
        answer="DSPy is a framework for programming language models by composing LM calls into pipelines and automatically optimizing prompts."
    ).with_inputs("question"),
    dspy.Example(
        question="What is RAG?",
        answer="RAG stands for Retrieval-Augmented Generation, which retrieves relevant documents and uses them as context for LLM to answer questions."
    ).with_inputs("question"),
    dspy.Example(
        question="What does ChainOfThought do in DSPy?",
        answer="ChainOfThought adds a reasoning field before the answer, forcing the LLM to think step by step."
    ).with_inputs("question"),
    dspy.Example(
        question="What is a metric function in DSPy?",
        answer="A metric function takes an example and a prediction, returning a score that optimizers use to evaluate prompt quality."
    ).with_inputs("question"),
    dspy.Example(
        question="What are DSPy optimizers?",
        answer="DSPy optimizers like BootstrapFewShot and MIPROv2 automatically improve prompts using training data."
    ).with_inputs("question"),
    dspy.Example(
        question="What is a DSPy Signature?",
        answer="DSPy Signatures define the input and output fields of a module and are used to generate prompts automatically."
    ).with_inputs("question"),
]

# ── 6. metric（简单关键词匹配，真实场景用 LLM 评分）──────────────
def answer_metric(example, prediction, trace=None):
    """检查预测答案是否包含标准答案的关键词"""
    gold_words = set(example.answer.lower().split())
    pred_words = set(prediction.answer.lower().split())
    overlap = len(gold_words & pred_words) / len(gold_words)
    return overlap >= 0.3  # 30% 关键词重叠即算正确

# ── 7. 优化前：打印一次 prompt ────────────────────────────────────
print("=" * 60)
print("【优化前】RAG prompt（无 few-shot）")
print("=" * 60)

rag = RAG()
result = rag(question="What is DSPy?")
print(lm.inspect_history(n=1))
print(f"\n>>> 检索到的 context:\n")
for i, p in enumerate(result.context, 1):
    print(f"  [{i}] {p[:80]}...")
print(f"\n>>> 答案: {result.answer}\n")

# ── 8. 用 BootstrapFewShot 优化 ──────────────────────────────────
print("=" * 60)
print("【优化中】BootstrapFewShot 自动挑选 few-shot demos ...")
print("=" * 60)
print()

optimizer = BootstrapFewShot(
    metric=answer_metric,
    max_bootstrapped_demos=2,
    max_labeled_demos=2,
)

optimized_rag = optimizer.compile(
    student=RAG(),
    trainset=trainset,
)

# ── 9. 优化后：打印 prompt（对比 few-shot 注入）──────────────────
print()
print("=" * 60)
print("【优化后】RAG prompt（注入了 few-shot demos）")
print("=" * 60)

result2 = optimized_rag(question="What is DSPy?")
print(lm.inspect_history(n=1))
print(f"\n>>> 答案: {result2.answer}\n")
