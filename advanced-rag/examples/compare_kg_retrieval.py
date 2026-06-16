"""
普通检索（BM25 + Dense）vs 知识图谱检索  Top-K 对比演示
==========================================================

知识图谱检索的核心优势在于三类问题：
  1. 多跳因果链  —— 普通检索漏掉中间环节
  2. 实体别名     —— 普通检索因词面不一致漏掉同义节点
  3. 组合约束     —— 普通检索返回零件，KG 直接返回满足约束的方案

本脚本：
  - 内嵌一个小型知识图谱（字典模拟，无需额外依赖）
  - 内嵌与图谱对应的平铺文档（供普通检索使用）
  - 用 BGE-small 做 Dense 检索、BM25 做稀疏检索
  - 打印 Top-K 召回结果对比表

依赖：sentence_transformers, rank_bm25, jieba, numpy
      （均已在 requirements.txt 中）
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import math
import re
import numpy as np
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import jieba

# ============================================================
# 1. 内嵌知识库：平铺文档（供 BM25 / Dense 检索）
# ============================================================
FLAT_DOCS = [
    # id, content
    ("D01", "bge-reranker-base 是 BAAI 发布的 cross-encoder 精排模型。"),
    ("D02", "cross-encoder 将 (query, doc) 拼接后送入模型，输出相关性打分，延迟比 bi-encoder 高。"),
    ("D03", "cross-encoder 每个 query-doc 对都要单独推理，query 有 K 个候选就要跑 K 次前向传播。"),
    ("D04", "K 次前向传播导致 RAG pipeline 的 reranker 阶段延迟随 K 线性增长，GPU 上约 10-200ms。"),
    ("D05", "降低 reranker 延迟的方案：减少候选 K、使用更小的模型（miniLM）、batch 推理、INT8 量化。"),
    ("D06", "INT8 量化可以让模型推理速度提升 2-4 倍，精度损失通常小于 1%。"),
    ("D07", "RAG pipeline 包括：文档加载、分块、向量化、检索、reranker、context 构建、LLM 生成。"),
    ("D08", "HyDE 是一种 query 改写方法，先让 LLM 生成假设答案，再用假设答案的 embedding 检索。"),
    ("D09", "step-back prompting 先问更抽象的背景问题，获取背景后再回答具体问题，也叫抽象问题法。"),
    ("D10", "step-back query 与 backoff question 是同一个概念的不同叫法。"),
    ("D11", "HyDE 的目标是让 embedding 空间里的 query 向量更接近真实文档。"),
    ("D12", "step-back prompting 的目标是通过降低问题粒度来扩大检索覆盖范围。"),
    ("D13", "中文电商 RAG 低延迟方案：embedding 用 bge-small-zh，reranker 可省略或用 miniLM。"),
    ("D14", "中文电商 RAG 低成本方案：不调 GPT-4，用开源模型；不用大 reranker，用 miniLM。"),
    ("D15", "中文电商 RAG 可追溯方案：在返回答案时附上来源文档 ID 和段落偏移量。"),
    ("D16", "同时满足低延迟+低成本+可追溯的推荐配置：bge-small-zh + miniLM(可选) + 来源引用。"),
    ("D17", "bge-small-zh 是专为中文优化的轻量 embedding 模型，推理速度快，中文语义理解好。"),
    ("D18", "miniLM 是微软发布的轻量 cross-encoder，体积小、速度快，适合对延迟敏感的场景。"),
]

# ============================================================
# 2. 内嵌知识图谱（邻接表，边含 label）
# ============================================================
# 格式：(源节点, 目标节点, 关系标签, 对应文档ID)
EDGES = [
    # 多跳链：bge-reranker → cross-encoder → K次推理 → 高延迟 → 优化方案
    ("bge-reranker-base", "cross-encoder",        "type_of",         "D01"),
    ("cross-encoder",     "K次前向传播",           "causes",          "D03"),
    ("K次前向传播",       "reranker延迟增长",       "causes",          "D04"),
    ("reranker延迟增长",  "降低延迟的方案",          "solved_by",       "D05"),
    ("降低延迟的方案",    "INT8量化",               "includes",        "D06"),
    # 别名链：step-back = backoff question
    ("step-back prompting", "backoff question",   "alias",           "D10"),
    ("step-back prompting", "抽象问题法",          "alias",           "D09"),
    ("HyDE",              "假设答案embedding检索",  "method",          "D08"),
    ("HyDE",              "接近真实文档",           "goal",            "D11"),
    ("step-back prompting","扩大检索覆盖",          "goal",            "D12"),
    # 组合约束链：中文电商 RAG 三约束 → 推荐配置
    ("低延迟",            "bge-small-zh",          "recommends",      "D13"),
    ("低成本",            "miniLM(可选)",           "recommends",      "D14"),
    ("可追溯",            "来源引用",               "recommends",      "D15"),
    ("中文电商RAG",       "低延迟",                 "constraint",      "D13"),
    ("中文电商RAG",       "低成本",                 "constraint",      "D14"),
    ("中文电商RAG",       "可追溯",                 "constraint",      "D15"),
    ("满足三约束的配置",  "中文电商RAG",             "satisfies",       "D16"),
    ("满足三约束的配置",  "bge-small-zh",           "component",       "D17"),
    ("满足三约束的配置",  "miniLM(可选)",           "component",       "D18"),
]

# 构建邻接表（双向查找）
from collections import defaultdict
_graph: dict[str, list] = defaultdict(list)  # node -> [(neighbor, label, doc_id)]
for src, dst, label, doc_id in EDGES:
    _graph[src].append((dst, label, doc_id))
    _graph[dst].append((src, f"←{label}", doc_id))  # 反向边也加入，方便双向遍历

# doc_id -> content 的映射
DOC_MAP = {did: content for did, content in FLAT_DOCS}

# ============================================================
# 3. 普通检索（BM25 + Dense）工具函数
# ============================================================

def bm25_tokenize(text: str) -> list[str]:
    return list(jieba.cut(text))

# 构建 BM25 索引
_corpus_ids = [did for did, _ in FLAT_DOCS]
_corpus_texts = [c for _, c in FLAT_DOCS]
_corpus_tokenized = [bm25_tokenize(t) for t in _corpus_texts]
_bm25 = BM25Okapi(_corpus_tokenized)

print("加载 BGE-small 向量模型（仅加载一次）...")
_bi_enc = SentenceTransformer("BAAI/bge-small-zh-v1.5")
_doc_vecs = _bi_enc.encode(_corpus_texts, normalize_embeddings=True, show_progress_bar=False)


def normal_retrieval(query: str, top_k: int = 5) -> list[tuple[str, float, str]]:
    """
    BM25 + Dense 混合检索（RRF 融合）
    返回：[(doc_id, rrf_score, content), ...]
    """
    # BM25
    q_tokens = bm25_tokenize(query)
    bm25_scores = _bm25.get_scores(q_tokens)
    bm25_ranks = np.argsort(bm25_scores)[::-1]

    # Dense
    q_vec = _bi_enc.encode(query, normalize_embeddings=True)
    dense_scores = _doc_vecs @ q_vec
    dense_ranks = np.argsort(dense_scores)[::-1]

    # RRF 融合 (k=60)
    rrf_scores: dict[int, float] = defaultdict(float)
    for rank, idx in enumerate(bm25_ranks):
        rrf_scores[idx] += 1.0 / (60 + rank + 1)
    for rank, idx in enumerate(dense_ranks):
        rrf_scores[idx] += 1.0 / (60 + rank + 1)

    sorted_indices = sorted(rrf_scores, key=lambda i: rrf_scores[i], reverse=True)[:top_k]
    return [(_corpus_ids[i], round(rrf_scores[i], 5), _corpus_texts[i]) for i in sorted_indices]


# ============================================================
# 4. KG 检索工具函数
# ============================================================

def _entity_match(query: str) -> list[str]:
    """
    从 query 里找出 KG 中存在的实体（简单字符串匹配）。
    工业上这里会用 NER 模型；这里用关键词匹配演示逻辑。
    """
    matched = []
    for node in _graph:
        if node in query or any(alias in query for alias in [node.replace("-", ""), node.lower()]):
            matched.append(node)
    # 额外的别名映射
    alias_map = {
        "reranker": "bge-reranker-base",
        "bge-reranker": "bge-reranker-base",
        "cross encoder": "cross-encoder",
        "step back": "step-back prompting",
        "step-back": "step-back prompting",
        "backoff": "step-back prompting",
        "Hyde": "HyDE",
        "hyde": "HyDE",
        "电商": "中文电商RAG",
        "低延迟": "低延迟",
        "低成本": "低成本",
        "可追溯": "可追溯",
    }
    for kw, node in alias_map.items():
        if kw in query and node not in matched:
            matched.append(node)
    return list(set(matched))


def kg_retrieval(query: str, hops: int = 2, top_k: int = 5) -> list[tuple[str, str, str, str]]:
    """
    KG 多跳检索：从匹配到的实体出发，BFS 遍历图谱 hops 步
    返回：[(path_desc, doc_id, content, hop), ...]  按路径展示
    """
    seed_entities = _entity_match(query)
    if not seed_entities:
        return []

    visited_docs: dict[str, tuple] = {}  # doc_id -> (path, hop)
    visited_nodes = set(seed_entities)
    frontier = [(e, [e], 0) for e in seed_entities]  # (node, path_list, hop)

    while frontier:
        node, path, hop = frontier.pop(0)
        if hop > hops:
            continue
        for neighbor, label, doc_id in _graph.get(node, []):
            if doc_id not in visited_docs:
                path_str = " → ".join(path + [f"[{label}]", neighbor])
                visited_docs[doc_id] = (path_str, hop + 1)
            if neighbor not in visited_nodes and hop < hops:
                visited_nodes.add(neighbor)
                frontier.append((neighbor, path + [f"[{label}]", neighbor], hop + 1))

    results = []
    for doc_id, (path_str, hop) in visited_docs.items():
        results.append((path_str, doc_id, DOC_MAP.get(doc_id, ""), hop))

    # 按 hop 排序（近的先）
    results.sort(key=lambda x: x[3])
    return results[:top_k]


# ============================================================
# 5. 测试案例
# ============================================================
TEST_CASES = [
    {
        "name": "案例1：多跳因果链  —— 「为什么用 bge-reranker 会让 RAG 延迟变高？怎么优化？」",
        "query": "为什么用 bge-reranker 会让 RAG 延迟变高？怎么优化？",
        "key_docs": ["D03", "D04", "D05"],  # 多跳中必须拿到的关键文档
        "explain": (
            "关键链：bge-reranker-base → cross-encoder → K次前向传播 → 延迟增长 → 优化方案\n"
            "普通检索：词面匹配 'reranker' 能拿到 D01/D07，但 D03(K次推理) 词面弱，容易被顶掉\n"
            "KG检索：沿边遍历，2跳内必然经过 D03 → D04 → D05，因果链完整"
        ),
    },
    {
        "name": "案例2：实体别名  —— 「step-back query 和 HyDE 的目标有什么不同？」",
        "query": "step-back query 和 HyDE 的目标有什么不同？",
        "key_docs": ["D09", "D10", "D11", "D12"],  # 含 alias 文档
        "explain": (
            "关键点：step-back 有别名 'backoff question'、'抽象问题法'\n"
            "普通检索：'step-back query' 词面命中 D09，但 D10(别名说明) 词面弱容易丢失\n"
            "KG检索：从 step-back prompting 出发，alias 边直接连到 D10，目标边连到 D12"
        ),
    },
    {
        "name": "案例3：组合约束  —— 「中文电商 RAG：低成本+低延迟+可追溯，推荐什么配置？」",
        "query": "中文电商 RAG：低成本、低延迟、可追溯，推荐什么配置？",
        "key_docs": ["D13", "D14", "D15", "D16"],  # 组合约束 → 最终配置
        "explain": (
            "关键点：三个约束都满足 → D16（推荐配置）是综合结论文档\n"
            "普通检索：D13/D14/D15 都有词面命中，但 D16 词面没有'低成本/低延迟/可追溯'等词，容易丢失\n"
            "KG检索：从'中文电商RAG'出发，1跳到'满足三约束的配置' → D16\n"
            "         2跳继续走到 bge-small-zh / miniLM 等组件节点 → D17/D18"
        ),
    },
]

# ============================================================
# 6. 主流程：打印对比表
# ============================================================

SEP = "=" * 78

def mark_key(doc_id: str, key_docs: list[str]) -> str:
    return "⭐" if doc_id in key_docs else "  "


def run_case(case: dict):
    query = case["query"]
    key_docs = case["key_docs"]

    print(f"\n{SEP}")
    print(f"  {case['name']}")
    print(SEP)
    print(f"  Query: {query}")
    print(f"\n  背景说明：")
    for line in case["explain"].split("\n"):
        print(f"    {line}")

    # ---- 普通检索 ----
    normal_results = normal_retrieval(query, top_k=5)
    normal_ids = [r[0] for r in normal_results]
    hit = sum(1 for d in key_docs if d in normal_ids)
    hit_rate = f"{hit}/{len(key_docs)}"

    print(f"\n  ── 普通检索（BM25 + Dense RRF）Top-5 ──  关键文档命中率: {hit_rate}")
    print(f"  {'序号':<4} {'DocID':<6} {'⭐':<3} {'RRF分':<8}  文档内容（前40字）")
    print(f"  {'-'*70}")
    for rank, (did, score, content) in enumerate(normal_results, 1):
        flag = mark_key(did, key_docs)
        print(f"  #{rank:<3} {did:<6} {flag:<3} {score:<8}  {content[:40]}")

    missing = [d for d in key_docs if d not in normal_ids]
    if missing:
        print(f"\n  ❌ 普通检索 未召回的关键文档: {missing}")
        for mid in missing:
            print(f"     {mid}: {DOC_MAP[mid]}")
    else:
        print(f"\n  ✅ 普通检索 关键文档全部命中")

    # ---- KG 检索 ----
    kg_results = kg_retrieval(query, hops=2, top_k=5)
    kg_ids = [r[1] for r in kg_results]
    hit_kg = sum(1 for d in key_docs if d in kg_ids)
    hit_rate_kg = f"{hit_kg}/{len(key_docs)}"
    seed_ents = _entity_match(query)

    print(f"\n  ── KG 检索（2跳图遍历）Top-5 ──  关键文档命中率: {hit_rate_kg}")
    print(f"  识别到的种子实体: {seed_ents}")
    print(f"  {'序号':<4} {'DocID':<6} {'⭐':<3} {'Hop':<5}  遍历路径  →  文档内容（前30字）")
    print(f"  {'-'*70}")
    if kg_results:
        for rank, (path_str, did, content, hop) in enumerate(kg_results, 1):
            flag = mark_key(did, key_docs)
            short_path = path_str[:45] + ("..." if len(path_str) > 45 else "")
            print(f"  #{rank:<3} {did:<6} {flag:<3} {hop}跳   {short_path}")
            print(f"  {'':>23}文档: {content[:38]}")
    else:
        print("  （未识别到 KG 中的种子实体，跳过 KG 检索）")

    missing_kg = [d for d in key_docs if d not in kg_ids]
    if missing_kg:
        print(f"\n  ❌ KG 检索 未召回的关键文档: {missing_kg}")
    else:
        print(f"\n  ✅ KG 检索 关键文档全部命中")


def main():
    print("\n" + SEP)
    print("  普通检索 vs 知识图谱检索  —  Top-K 召回对比")
    print("  ⭐ = 该案例的关键文档（多跳问题必须召回才能答对）")
    print(SEP)

    for case in TEST_CASES:
        run_case(case)

    print(f"\n{SEP}")
    print("  总结")
    print(SEP)
    print("""
  问题类型          普通检索（BM25+Dense）                 KG检索
  ──────────────────────────────────────────────────────────────────
  多跳因果链   词面弱的"中间环节"文档排名靠后/丢失      沿边强制经过中间节点
  实体别名     别名文档词面不一致时容易漏掉              alias 边聚合所有同义写法
  组合约束     能召回各约束文档，但"结论文档"无关键词    从约束出发走到结论节点

  工业上什么时候值得建 KG：
    - 知识有明确的因果/归属/组合关系（产品 BOM、医疗诊断链、法规引用）
    - 实体别名多（药名/商品名/缩写）
    - 需要多跳推理（A 影响 B，B 影响 C，问 A 对 C 的影响）

  什么时候普通检索足够：
    - 知识相对独立，每段文档可以单独回答问题
    - 问题是单跳的（"X 是什么"、"X 有什么特点"）
    - 知识更新频率高（KG 维护成本高）
""")


if __name__ == "__main__":
    main()
