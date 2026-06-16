"""
混合检索器模块

实现 BM25（稀疏检索）+ Dense Embedding（稠密检索）的混合检索策略。

混合检索的优势：
- BM25: 精确关键词匹配，处理专有名词、缩写
- Dense: 语义理解，处理同义词、改写、上下文
"""
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import Chroma
from rank_bm25 import BM25Okapi
import jieba
import numpy as np

from .config import config


@dataclass
class RetrievalResult:
    """检索结果"""
    document: Document
    score: float
    source: str  # "bm25", "dense", "hybrid"


class BM25Retriever:
    """
    BM25 稀疏检索器
    
    基于词频的经典检索算法，擅长精确匹配。
    """
    
    # 类变量：标记是否已加载自定义词典
    _dict_loaded = False
    
    @classmethod
    def load_custom_dict(cls, dict_path: str = None):
        """加载自定义词典（只需加载一次）"""
        if cls._dict_loaded:
            return
        
        if dict_path is None:
            # 默认词典路径
            dict_path = Path(__file__).parent.parent / "data" / "custom_dict.txt"
        
        if Path(dict_path).exists():
            jieba.load_userdict(str(dict_path))
            print(f"[BM25] 已加载自定义词典: {dict_path}")
        
        cls._dict_loaded = True
    
    def __init__(self, documents: List[Document], use_jieba: bool = True, custom_dict: str = None):
        """
        Args:
            documents: 文档列表
            use_jieba: 是否使用结巴分词（中文推荐开启）
            custom_dict: 自定义词典路径
        """
        self.documents = documents
        self.use_jieba = use_jieba
        
        # 加载自定义词典
        if use_jieba:
            self.load_custom_dict(custom_dict)
        
        # 分词
        self.tokenized_docs = [self._tokenize(doc.page_content) for doc in documents]
        
        # 构建 BM25 索引
        self.bm25 = BM25Okapi(self.tokenized_docs)
    
    def _tokenize(self, text: str) -> List[str]:
        """分词"""
        if self.use_jieba:
            return list(jieba.cut(text))
        else:
            # 简单的空格分词
            return text.lower().split()
    
    def retrieve(self, query: str, top_k: int = 10) -> List[RetrievalResult]:
        """
        检索最相关的文档
        
        Args:
            query: 查询文本
            top_k: 返回数量
        
        Returns:
            检索结果列表
        """
        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        
        # 获取 top_k 的索引
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # 只返回有分数的
                results.append(RetrievalResult(
                    document=self.documents[idx],
                    score=float(scores[idx]),
                    source="bm25"
                ))
        
        return results


class DenseRetriever:
    """
    Dense 稠密检索器
    
    基于向量相似度的检索，擅长语义匹配。
    """
    
    def __init__(
        self, 
        embeddings: Embeddings,
        persist_directory: Optional[str] = None,
        collection_name: str = "dense_retriever"
    ):
        self.embeddings = embeddings
        self.persist_directory = persist_directory or config.vector_store.persist_directory
        self.collection_name = collection_name
        self.vectorstore: Optional[Chroma] = None
    
    def add_documents(self, documents: List[Document]):
        """添加文档到向量库"""
        if self.vectorstore is None:
            self.vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=self.persist_directory,
                collection_name=self.collection_name,
            )
        else:
            self.vectorstore.add_documents(documents)
    
    def load(self):
        """从持久化目录加载向量库"""
        self.vectorstore = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings,
            collection_name=self.collection_name,
        )
    
    def retrieve(self, query: str, top_k: int = 10) -> List[RetrievalResult]:
        """检索最相关的文档"""
        if self.vectorstore is None:
            raise ValueError("向量库未初始化，请先调用 add_documents 或 load")
        
        # 使用带分数的相似度搜索
        results_with_scores = self.vectorstore.similarity_search_with_relevance_scores(
            query, k=top_k
        )
        
        return [
            RetrievalResult(
                document=doc,
                score=score,
                source="dense"
            )
            for doc, score in results_with_scores
        ]


class HybridRetriever:
    """
    混合检索器
    
    结合 BM25 和 Dense 检索的结果，使用加权融合。
    
    融合策略：
    1. RRF (Reciprocal Rank Fusion) - 基于排名的融合
    2. 线性加权 - 基于分数的融合
    """
    
    def __init__(
        self,
        embeddings: Embeddings,
        bm25_weight: float = None,
        dense_weight: float = None,
        fusion_method: str = "rrf"  # "rrf" 或 "linear"
    ):
        self.embeddings = embeddings
        self.bm25_weight = bm25_weight or config.retrieval.bm25_weight
        self.dense_weight = dense_weight or config.retrieval.dense_weight
        self.fusion_method = fusion_method
        
        self.bm25_retriever: Optional[BM25Retriever] = None
        self.dense_retriever: Optional[DenseRetriever] = None
        
        # 用于 Sentence Window 的父块映射
        self.parent_mapping: Dict[str, Document] = {}
    
    def add_documents(
        self, 
        documents: List[Document],
        parent_mapping: Optional[Dict[str, Document]] = None
    ):
        """
        添加文档到检索器
        
        Args:
            documents: 文档列表（小块，用于检索）
            parent_mapping: chunk_id -> parent_chunk 的映射（用于 Sentence Window）
        """
        if not documents:
            raise ValueError("文档列表为空，无法构建索引")
        
        # BM25 检索器
        self.bm25_retriever = BM25Retriever(documents)
        
        # Dense 检索器
        self.dense_retriever = DenseRetriever(
            embeddings=self.embeddings,
            collection_name=config.vector_store.collection_name
        )
        self.dense_retriever.add_documents(documents)
        
        # 保存父块映射
        if parent_mapping:
            self.parent_mapping = parent_mapping
    
    def retrieve(
        self, 
        query: str, 
        top_k: int = None,
        return_parent: bool = True
    ) -> List[RetrievalResult]:
        """
        混合检索
        
        Args:
            query: 查询文本
            top_k: 返回数量
            return_parent: 是否返回父块（Sentence Window）
        
        Returns:
            融合后的检索结果
        """
        top_k = top_k or config.retrieval.top_k
        
        if self.bm25_retriever is None or self.dense_retriever is None:
            raise ValueError("检索器未初始化，请先调用 add_documents")
        
        # BM25 检索
        bm25_results = self.bm25_retriever.retrieve(query, top_k=top_k)
        
        # Dense 检索
        dense_results = self.dense_retriever.retrieve(query, top_k=top_k)
        
        # 融合结果
        if self.fusion_method == "rrf":
            fused_results = self._rrf_fusion(bm25_results, dense_results, top_k)
        else:
            fused_results = self._linear_fusion(bm25_results, dense_results, top_k)
        
        # 如果启用 Sentence Window，替换为父块
        if return_parent and self.parent_mapping:
            fused_results = self._expand_to_parent(fused_results)
        
        return fused_results
    
    def _rrf_fusion(
        self,
        bm25_results: List[RetrievalResult],
        dense_results: List[RetrievalResult],
        top_k: int,
        k: int = 60  # RRF 参数
    ) -> List[RetrievalResult]:
        """
        Reciprocal Rank Fusion (RRF) 融合
        
        RRF Score = Σ 1/(k + rank)
        
        优点：不依赖分数的绝对值，只依赖排名
        """
        doc_scores: Dict[str, float] = {}
        doc_map: Dict[str, Document] = {}
        
        # BM25 结果
        for rank, result in enumerate(bm25_results):
            doc_id = self._get_doc_id(result.document)
            rrf_score = self.bm25_weight / (k + rank + 1)
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + rrf_score
            doc_map[doc_id] = result.document
        
        # Dense 结果
        for rank, result in enumerate(dense_results):
            doc_id = self._get_doc_id(result.document)
            rrf_score = self.dense_weight / (k + rank + 1)
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + rrf_score
            doc_map[doc_id] = result.document
        
        # 排序
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        return [
            RetrievalResult(
                document=doc_map[doc_id],
                score=score,
                source="hybrid"
            )
            for doc_id, score in sorted_docs
        ]
    
    def _linear_fusion(
        self,
        bm25_results: List[RetrievalResult],
        dense_results: List[RetrievalResult],
        top_k: int
    ) -> List[RetrievalResult]:
        """
        线性加权融合
        
        Final Score = α * norm(bm25_score) + (1-α) * norm(dense_score)
        """
        doc_scores: Dict[str, Tuple[float, float]] = {}  # (bm25_score, dense_score)
        doc_map: Dict[str, Document] = {}
        
        # 收集分数
        for result in bm25_results:
            doc_id = self._get_doc_id(result.document)
            doc_scores[doc_id] = (result.score, doc_scores.get(doc_id, (0, 0))[1])
            doc_map[doc_id] = result.document
        
        for result in dense_results:
            doc_id = self._get_doc_id(result.document)
            doc_scores[doc_id] = (doc_scores.get(doc_id, (0, 0))[0], result.score)
            doc_map[doc_id] = result.document
        
        # 归一化
        bm25_scores = [s[0] for s in doc_scores.values()]
        dense_scores = [s[1] for s in doc_scores.values()]
        
        bm25_max = max(bm25_scores) if bm25_scores else 1
        dense_max = max(dense_scores) if dense_scores else 1
        
        # 计算融合分数
        final_scores = {}
        for doc_id, (bm25_s, dense_s) in doc_scores.items():
            norm_bm25 = bm25_s / bm25_max if bm25_max > 0 else 0
            norm_dense = dense_s / dense_max if dense_max > 0 else 0
            final_scores[doc_id] = self.bm25_weight * norm_bm25 + self.dense_weight * norm_dense
        
        # 排序
        sorted_docs = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        return [
            RetrievalResult(
                document=doc_map[doc_id],
                score=score,
                source="hybrid"
            )
            for doc_id, score in sorted_docs
        ]
    
    def _expand_to_parent(self, results: List[RetrievalResult]) -> List[RetrievalResult]:
        """
        将小块扩展到父块 (Sentence Window Retrieval)
        
        检索时用小块精准匹配，返回时用大块保证上下文完整
        """
        expanded_results = []
        seen_parent_ids = set()
        
        for result in results:
            chunk_id = result.document.metadata.get("chunk_id")
            parent_id = result.document.metadata.get("parent_id")
            
            if parent_id and parent_id not in seen_parent_ids:
                if chunk_id in self.parent_mapping:
                    expanded_results.append(RetrievalResult(
                        document=self.parent_mapping[chunk_id],
                        score=result.score,
                        source=result.source
                    ))
                    seen_parent_ids.add(parent_id)
            elif not parent_id:
                # 没有父块映射，直接使用原文档
                expanded_results.append(result)
        
        return expanded_results
    
    @staticmethod
    def _get_doc_id(doc: Document) -> str:
        """获取文档唯一标识"""
        return doc.metadata.get("chunk_id") or hash(doc.page_content)


if __name__ == "__main__":
    # 测试混合检索
    from .document_loader import SmartChunker
    from .embeddings import get_local_embeddings
    
    # 创建测试文档
    test_docs = [
        Document(page_content="RAG 是检索增强生成技术，结合了检索和生成的优点。", metadata={"source": "doc1"}),
        Document(page_content="BM25 是一种经典的稀疏检索算法，基于词频统计。", metadata={"source": "doc2"}),
        Document(page_content="Dense Retrieval 使用神经网络将文本编码为向量进行检索。", metadata={"source": "doc3"}),
        Document(page_content="混合检索结合了 BM25 和向量检索的优势。", metadata={"source": "doc4"}),
        Document(page_content="LangChain 是一个用于构建 LLM 应用的框架。", metadata={"source": "doc5"}),
    ]
    
    try:
        embeddings = get_local_embeddings()
        
        # 创建混合检索器
        retriever = HybridRetriever(embeddings=embeddings)
        retriever.add_documents(test_docs)
        
        # 测试检索
        query = "什么是 RAG 技术？"
        results = retriever.retrieve(query, top_k=3)
        
        print(f"查询: {query}\n")
        for i, result in enumerate(results):
            print(f"{i+1}. [Score: {result.score:.4f}] {result.document.page_content[:50]}...")
            
    except Exception as e:
        print(f"测试失败: {e}")
