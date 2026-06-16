# Advanced RAG 技术详解

## 为什么需要 Advanced RAG？

基础的 Naive RAG 存在以下问题：

1. **召回质量不稳定**：简单的向量检索可能漏掉重要信息或引入无关内容
2. **上下文碎片化**：固定大小的分块可能破坏语义完整性
3. **查询理解不足**：用户的原始查询可能表达不清或有歧义
4. **排序不够精准**：仅靠向量相似度难以准确判断相关性

## Advanced RAG 的核心技术

### 1. 智能分块策略

#### Sentence Window Retrieval
核心思想：用小块做精准检索，返回大块保证上下文完整。

```python
# 检索时使用小块（更精准）
small_chunks = split_text(doc, chunk_size=256)
# 返回时扩展到大块（更完整）
parent_chunks = split_text(doc, chunk_size=1024)
```

#### 语义分块
根据语义边界（而非固定长度）进行分块：
- 计算相邻句子的相似度
- 在相似度低于阈值处切分
- 保持语义的连贯性

### 2. 混合检索

结合稀疏检索和稠密检索的优势：

| 检索方式 | 优势 | 劣势 |
|---------|------|------|
| BM25（稀疏）| 精确匹配关键词、处理专有名词 | 无法理解语义相似 |
| Dense（稠密）| 语义理解、同义词匹配 | 可能漏掉精确匹配 |

融合公式：
```
final_score = α * bm25_score + (1-α) * dense_score
```

或使用 RRF（Reciprocal Rank Fusion）：
```
rrf_score = Σ 1/(k + rank)
```

### 3. Query 改写技术

#### HyDE（Hypothetical Document Embeddings）
让 LLM 先生成假设性答案，用答案的 embedding 去检索。

原理：假设答案的语义和真实文档更接近。

```
Query: "什么是RAG?"
↓ LLM生成假设答案
HyDE: "RAG（检索增强生成）是一种结合信息检索与文本生成的技术..."
↓ 用HyDE的embedding检索
Results: [相关文档1, 相关文档2, ...]
```

#### Multi-Query
从不同角度生成多个查询，合并检索结果：
- 增加召回的多样性
- 覆盖用户可能的不同表达

#### Step-Back Prompting
先问更宽泛的问题，获取背景知识后再回答具体问题。

### 4. Reranker 重排序

#### Bi-encoder vs Cross-encoder

| 类型 | 工作方式 | 速度 | 精度 |
|-----|---------|------|------|
| Bi-encoder | 分别编码 query 和 doc | 快 | 中 |
| Cross-encoder | 同时编码 (query, doc) 对 | 慢 | 高 |

推荐流程：
1. 用 Bi-encoder 快速召回 Top-100
2. 用 Cross-encoder 精排 Top-10
3. 返回 Top-3 作为最终结果

常用 Reranker 模型：
- `BAAI/bge-reranker-base`
- `BAAI/bge-reranker-large`
- `ms-marco-MiniLM-L-12-v2`

### 5. 上下文压缩

在将检索结果送入 LLM 之前，进行压缩和过滤：

1. **相关性过滤**：移除分数低于阈值的文档
2. **内容提取**：只保留与问题相关的段落
3. **去重**：移除重复或高度相似的内容
4. **排序**：按相关性从高到低排列

## Self-RAG 和 CRAG

### Self-RAG
让模型自己判断是否需要检索：
1. 生成初步答案
2. 判断答案是否需要外部知识支持
3. 如果需要，执行检索并重新生成
4. 验证检索结果是否有帮助

### CRAG（Corrective RAG）
检索结果的自我纠正：
1. 评估检索文档的相关性
2. 如果相关性低，进行网络搜索补充
3. 如果部分相关，提取有用片段
4. 基于高质量上下文生成答案

## 评测指标

### 检索评测
- **召回率（Recall）**：相关文档被检索到的比例
- **精确率（Precision）**：检索结果中相关文档的比例
- **MRR（Mean Reciprocal Rank）**：第一个相关结果的排名倒数的平均值
- **NDCG**：考虑排序位置的评分指标

### 生成评测
- **Faithfulness**：答案是否忠实于检索内容
- **Answer Relevancy**：答案是否回答了问题
- **Context Relevancy**：检索内容是否与问题相关

### 常用评测框架
- RAGAS：专门针对 RAG 的评测框架
- TruLens：LLM 应用的评测和监控
