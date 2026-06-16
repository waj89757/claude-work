"""
对比 Bi-encoder 和 Cross-encoder 的排序效果

演示用同一组 (query, docs)：
- Bi-encoder：各自算向量 → cosine 相似度排序
- Cross-encoder：(query, doc) 拼在一起 → 直接输出相关性分数排序
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder

# ============================================================
# 测试数据（4 个场景，每个都是 bi-encoder 容易翻车的）
# ============================================================
test_cases = [
    {
        "name": "场景A：同主题但没回答问题",
        "query": "RAG 怎么减少幻觉？",
        "docs": [
            "RAG 通过在生成前检索真实文档作为上下文，从而大幅减少模型编造信息，降低幻觉发生率。",
            "RAG 的整体架构包括文档加载、向量库、检索器和生成模型四个部分。",
            "Fine-tuning 可以让模型学习特定领域的语言风格和输出格式。",
        ],
        "correct": 0,  # 第0个文档是正确答案
    },
    {
        "name": "场景B：否定语义，问'不适合'",
        "query": "RAG 不适合什么场景？",
        "docs": [
            "RAG 非常适合知识需要频繁更新的场景，以及需要引用来源的场合。",
            "当系统对延迟要求极高时，HyDE 和 Reranker 等步骤会显著增加耗时，不建议使用复杂 RAG。",
            "RAG 和 Fine-tuning 各有优劣，要根据具体需求选择。",
        ],
        "correct": 1,
    },
    {
        "name": "场景C：问具体数字",
        "query": "混合检索的 BM25 权重默认是多少？",
        "docs": [
            "混合检索结合了 BM25 和 Dense 向量检索，融合后返回 Top-K 个文档。",
            "在配置文件中，BM25_WEIGHT=0.3，即 BM25 占 30%，Dense 占 70%。",
            "RRF 融合算法不依赖分数的绝对值，只依赖排名。",
        ],
        "correct": 1,
    },
    {
        "name": "场景D：问'区别/对比'",
        "query": "RAG 和 Fine-tuning 的区别是什么？",
        "docs": [
            "RAG 是检索增强生成，在生成前先检索相关文档，适合知识频繁更新的场景。",
            "Fine-tuning 是对预训练模型继续训练，让模型学习新的知识或行为模式。",
            "RAG 和 Fine-tuning 的核心区别：RAG 不改变模型参数，知识在外部；Fine-tuning 修改参数，知识固化在模型内。应该选哪个取决于知识是否需要频繁更新。",
        ],
        "correct": 2,
    },
]

# ============================================================
# 加载模型
# ============================================================
print("加载 Bi-encoder 模型（BGE-small，用于向量检索）...")
bi_encoder = SentenceTransformer("BAAI/bge-small-zh-v1.5")

print("加载 Cross-encoder 模型（BGE-reranker-base，用于重排序）...")
cross_encoder = CrossEncoder("BAAI/bge-reranker-base")

print("\n" + "=" * 70)

# ============================================================
# 对每个场景分别跑
# ============================================================
for case in test_cases:
    query = case["query"]
    docs = case["docs"]
    correct_idx = case["correct"]

    print(f"\n【{case['name']}】")
    print(f"Query: {query}\n")

    # ---------- Bi-encoder ----------
    query_vec = bi_encoder.encode(query, normalize_embeddings=True)
    doc_vecs = bi_encoder.encode(docs, normalize_embeddings=True)
    bi_scores = [float(np.dot(query_vec, dv)) for dv in doc_vecs]
    bi_ranking = sorted(range(len(docs)), key=lambda i: bi_scores[i], reverse=True)

    # ---------- Cross-encoder ----------
    pairs = [(query, doc) for doc in docs]
    ce_scores = cross_encoder.predict(pairs)
    ce_ranking = sorted(range(len(docs)), key=lambda i: ce_scores[i], reverse=True)

    # ---------- 打印对比 ----------
    print(f"{'Doc':<4} {'文档内容（前30字）':<32} {'Bi-score':>9} {'Bi排名':>6} {'CE-score':>10} {'CE排名':>6} {'正确答案':>8}")
    print("-" * 88)
    for i, doc in enumerate(docs):
        is_correct = "✅" if i == correct_idx else "  "
        bi_rank = bi_ranking.index(i) + 1
        ce_rank = ce_ranking.index(i) + 1
        print(f"Doc{i}  {doc[:30]:<32} {bi_scores[i]:>9.4f} {bi_rank:>6} {ce_scores[i]:>10.4f} {ce_rank:>6} {is_correct:>8}")

    bi_correct_rank = bi_ranking.index(correct_idx) + 1
    ce_correct_rank = ce_ranking.index(correct_idx) + 1
    bi_result = "✅" if bi_correct_rank == 1 else f"❌ 排第{bi_correct_rank}"
    ce_result = "✅" if ce_correct_rank == 1 else f"❌ 排第{ce_correct_rank}"
    print(f"\n  → Bi-encoder 把正确答案排在: {bi_result}")
    print(f"  → Cross-encoder 把正确答案排在: {ce_result}")
    print("=" * 70)

print("\n✅ 对比完成！")
print("\n关键规律：")
print("- Bi-encoder 容易被'同主题但没回答问题'的 doc 迷惑（向量距离近）")
print("- Cross-encoder 能更好地理解'否定语义、数字约束、对比意图'等细节")
