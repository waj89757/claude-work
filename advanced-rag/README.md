# LangChain Advanced RAG 实践项目

这是一个完整的 Advanced RAG（Retrieval-Augmented Generation）实现，包含工业级 RAG 的核心技术：

## 🏗️ 架构特点

```
Query → Query改写/扩展 → 混合检索(BM25+Dense) → Reranker重排序 → LLM生成
```

### 核心功能

1. **智能文档分块** - Sentence-window + Parent-document 策略
2. **混合检索** - BM25（稀疏）+ Dense Embedding（稠密）
3. **Query 改写** - HyDE（Hypothetical Document Embeddings）
4. **重排序** - Cross-encoder Reranker
5. **上下文压缩** - 过滤无关内容

## 📁 项目结构

```
advanced-rag/
├── requirements.txt      # 依赖
├── .env.example          # 环境变量示例
├── src/
│   ├── __init__.py
│   ├── document_loader.py    # 文档加载与分块
│   ├── embeddings.py         # Embedding 封装
│   ├── retriever.py          # 混合检索器
│   ├── reranker.py           # 重排序器
│   ├── query_transform.py    # Query 改写
│   ├── rag_pipeline.py       # 完整 Pipeline
│   └── config.py             # 配置管理
├── data/
│   └── sample_docs/          # 示例文档
├── examples/
│   └── demo.py               # 运行示例
└── README.md
```

## 🚀 快速开始

### 1. 安装依赖

```bash
cd advanced-rag
pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env 文件，填入你的 API Key
```

### 3. 运行示例

```bash
python examples/demo.py
```

## 🔧 核心技术说明

### 1. 智能分块（Chunking）

使用 **Sentence Window Retrieval**：
- 小块用于精准检索（更高召回率）
- 返回时扩展到大块（更完整上下文）

### 2. 混合检索（Hybrid Retrieval）

```python
final_score = α * bm25_score + (1-α) * dense_score
```

- **BM25**: 关键词精确匹配，处理专有名词
- **Dense**: 语义相似度，处理同义词/改写

### 3. HyDE Query 改写

让 LLM 先生成假设性答案，用答案的 embedding 去检索：

```
Query: "什么是RAG?"
↓ LLM生成假设答案
HyDE: "RAG是检索增强生成技术，结合检索和生成..."
↓ 用HyDE文本做embedding检索
```

### 4. Reranker 重排序

Cross-encoder 对 (query, document) 对精细打分，比 bi-encoder 更准确。

## 📊 对比：Naive RAG vs Advanced RAG

| 组件 | Naive RAG | 本项目 Advanced RAG |
|------|-----------|---------------------|
| 分块 | 固定512 tokens | Sentence-window |
| 检索 | 单路向量 | BM25 + Dense |
| Query | 直接使用 | HyDE 改写 |
| 排序 | cosine similarity | Cross-encoder Rerank |
| 过滤 | 无 | 相关性阈值过滤 |

## 🎯 学习路径

1. **理解基础**: 先看 `src/rag_pipeline.py` 了解完整流程
2. **深入组件**: 逐个阅读 retriever、reranker 等模块
3. **动手实验**: 修改参数，对比效果
4. **进阶**: 添加更多功能（GraphRAG、Self-RAG）

## 📚 参考资料

- [LangChain RAG Tutorial](https://python.langchain.com/docs/tutorials/rag/)
- [HyDE Paper](https://arxiv.org/abs/2212.10496)
- [Sentence Window Retrieval](https://docs.llamaindex.ai/en/stable/examples/node_postprocessor/MetadataReplacementDemo/)
