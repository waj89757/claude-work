"""
重排序模块

实现多种 Reranker 策略：
1. Cross-encoder Reranker - 使用交叉编码器精排
2. LLM Reranker - 使用 LLM 作为评判
3. Cohere Reranker - 使用 Cohere API

为什么需要 Reranker？
- Bi-encoder (Dense Retrieval): 分别编码 query 和 doc，计算相似度，速度快但精度有限
- Cross-encoder: 同时编码 (query, doc) 对，直接输出相关性分数，精度高但速度慢
- 先用 Bi-encoder 召回大量候选，再用 Cross-encoder 精排 Top-K
"""
from typing import List, Optional
from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from .config import config


@dataclass
class RerankResult:
    """重排序结果"""
    document: Document
    score: float
    original_rank: int
    new_rank: int


class CrossEncoderReranker:
    """
    Cross-encoder 重排序器
    
    使用 FlashRank 或 sentence-transformers 的 Cross-encoder 模型
    """
    
    def __init__(self, model_name: str = "ms-marco-MiniLM-L-12-v2"):
        """
        Args:
            model_name: Cross-encoder 模型名称
                - "ms-marco-MiniLM-L-12-v2" (快速，英文)
                - "BAAI/bge-reranker-base" (中英文)
                - "BAAI/bge-reranker-large" (更准确)
        """
        self.model_name = model_name
        self._reranker = None
    
    def _init_flashrank(self):
        """初始化 FlashRank（轻量级 Reranker）"""
        try:
            from flashrank import Ranker, RerankRequest
            self._reranker = Ranker(model_name=self.model_name)
            self._use_flashrank = True
        except ImportError:
            self._use_flashrank = False
    
    def _init_cross_encoder(self):
        """初始化 sentence-transformers Cross-encoder"""
        try:
            from sentence_transformers import CrossEncoder
            self._reranker = CrossEncoder(self.model_name)
            self._use_flashrank = False
        except ImportError:
            raise ImportError(
                "请安装 flashrank 或 sentence-transformers: "
                "pip install flashrank 或 pip install sentence-transformers"
            )
    
    def rerank(
        self, 
        query: str, 
        documents: List[Document], 
        top_k: int = None
    ) -> List[RerankResult]:
        """
        重排序文档
        
        Args:
            query: 查询文本
            documents: 待排序的文档列表
            top_k: 返回数量
        
        Returns:
            重排序后的结果
        """
        if not documents:
            return []
        
        top_k = top_k or config.retrieval.rerank_top_k
        
        # 延迟初始化
        if self._reranker is None:
            self._init_flashrank()
            if not self._use_flashrank:
                self._init_cross_encoder()
        
        if self._use_flashrank:
            return self._rerank_flashrank(query, documents, top_k)
        else:
            return self._rerank_cross_encoder(query, documents, top_k)
    
    def _rerank_flashrank(
        self, 
        query: str, 
        documents: List[Document], 
        top_k: int
    ) -> List[RerankResult]:
        """使用 FlashRank 重排序"""
        from flashrank import RerankRequest
        
        # 准备输入
        passages = [
            {"id": i, "text": doc.page_content}
            for i, doc in enumerate(documents)
        ]
        
        request = RerankRequest(query=query, passages=passages)
        results = self._reranker.rerank(request)
        
        # 转换结果
        reranked = []
        for new_rank, result in enumerate(results[:top_k]):
            original_rank = result["id"]
            reranked.append(RerankResult(
                document=documents[original_rank],
                score=result["score"],
                original_rank=original_rank,
                new_rank=new_rank
            ))
        
        return reranked
    
    def _rerank_cross_encoder(
        self, 
        query: str, 
        documents: List[Document], 
        top_k: int
    ) -> List[RerankResult]:
        """使用 sentence-transformers Cross-encoder 重排序"""
        # 准备输入对
        pairs = [(query, doc.page_content) for doc in documents]
        
        # 计算分数
        scores = self._reranker.predict(pairs)
        
        # 排序
        scored_docs = list(zip(range(len(documents)), documents, scores))
        scored_docs.sort(key=lambda x: x[2], reverse=True)
        
        # 转换结果
        reranked = []
        for new_rank, (original_rank, doc, score) in enumerate(scored_docs[:top_k]):
            reranked.append(RerankResult(
                document=doc,
                score=float(score),
                original_rank=original_rank,
                new_rank=new_rank
            ))
        
        return reranked


class LLMReranker:
    """
    LLM 重排序器
    
    使用 LLM 作为评判，对每个文档评分。
    适合需要复杂理解的场景，但速度较慢。
    """
    
    def __init__(self, llm: Optional[ChatOpenAI] = None):
        self.llm = llm or ChatOpenAI(
            model=config.llm.model_name,
            temperature=0,
            api_key=config.llm.api_key,
            base_url=config.llm.base_url,
        )
        
        self.prompt = ChatPromptTemplate.from_template("""评估以下文档与查询的相关性。

查询: {query}

文档: {document}

请给出1-10的相关性分数（10最相关），只输出数字:""")
    
    def rerank(
        self, 
        query: str, 
        documents: List[Document], 
        top_k: int = None
    ) -> List[RerankResult]:
        """使用 LLM 重排序"""
        if not documents:
            return []
        
        top_k = top_k or config.retrieval.rerank_top_k
        
        # 对每个文档评分
        scored_docs = []
        for i, doc in enumerate(documents):
            try:
                response = self.llm.invoke(
                    self.prompt.format(query=query, document=doc.page_content[:500])
                )
                score = float(response.content.strip())
            except:
                score = 0.0
            
            scored_docs.append((i, doc, score))
        
        # 排序
        scored_docs.sort(key=lambda x: x[2], reverse=True)
        
        # 转换结果
        reranked = []
        for new_rank, (original_rank, doc, score) in enumerate(scored_docs[:top_k]):
            reranked.append(RerankResult(
                document=doc,
                score=score / 10.0,  # 归一化到 0-1
                original_rank=original_rank,
                new_rank=new_rank
            ))
        
        return reranked


class Reranker:
    """
    统一的 Reranker 接口
    
    根据配置自动选择合适的实现
    """
    
    def __init__(self, method: str = "cross_encoder"):
        """
        Args:
            method: 重排序方法
                - "cross_encoder": 使用 Cross-encoder（推荐）
                - "llm": 使用 LLM
        """
        self.method = method
        
        if method == "cross_encoder":
            self._reranker = CrossEncoderReranker()
        elif method == "llm":
            self._reranker = LLMReranker()
        else:
            raise ValueError(f"不支持的重排序方法: {method}")
    
    def rerank(
        self, 
        query: str, 
        documents: List[Document], 
        top_k: int = None
    ) -> List[RerankResult]:
        """重排序文档"""
        return self._reranker.rerank(query, documents, top_k)


def filter_by_relevance(
    results: List[RerankResult],
    threshold: float = None
) -> List[RerankResult]:
    """
    根据相关性阈值过滤结果
    
    Args:
        results: 重排序结果
        threshold: 相关性阈值 (0-1)
    
    Returns:
        过滤后的结果
    """
    threshold = threshold or config.retrieval.relevance_threshold
    return [r for r in results if r.score >= threshold]


if __name__ == "__main__":
    # 测试 Reranker
    test_docs = [
        Document(page_content="RAG 结合了检索和生成，是一种有效的知识增强方法。"),
        Document(page_content="今天天气很好，适合出去散步。"),
        Document(page_content="检索增强生成技术可以减少大模型的幻觉问题。"),
        Document(page_content="Python 是一种流行的编程语言。"),
        Document(page_content="RAG 的核心是在生成前检索相关文档作为上下文。"),
    ]
    
    query = "什么是 RAG 技术？"
    
    print(f"查询: {query}\n")
    print("=== 原始顺序 ===")
    for i, doc in enumerate(test_docs):
        print(f"{i+1}. {doc.page_content[:40]}...")
    
    try:
        reranker = Reranker(method="cross_encoder")
        results = reranker.rerank(query, test_docs, top_k=3)
        
        print("\n=== 重排序后 ===")
        for result in results:
            print(f"[Score: {result.score:.4f}] (原排名 {result.original_rank+1} -> {result.new_rank+1}) "
                  f"{result.document.page_content[:40]}...")
    except Exception as e:
        print(f"\nCross-encoder 测试失败: {e}")
        print("尝试 LLM Reranker...")
        
        reranker = Reranker(method="llm")
        results = reranker.rerank(query, test_docs, top_k=3)
        
        print("\n=== LLM 重排序后 ===")
        for result in results:
            print(f"[Score: {result.score:.4f}] {result.document.page_content[:40]}...")
