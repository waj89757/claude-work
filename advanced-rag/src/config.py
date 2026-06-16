"""
Advanced RAG 配置管理
"""
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()


@dataclass
class EmbeddingConfig:
    """Embedding 配置"""
    model_name: str = field(default_factory=lambda: os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"))
    # 是否使用本地模型
    use_local: bool = field(default_factory=lambda: os.getenv("USE_LOCAL_EMBEDDING", "false").lower() == "true")
    local_model_name: str = "BAAI/bge-small-zh-v1.5"


@dataclass
class LLMConfig:
    """LLM 配置"""
    model_name: str = field(default_factory=lambda: os.getenv("LLM_MODEL", "gpt-4o-mini"))
    api_key: Optional[str] = field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))
    base_url: Optional[str] = field(default_factory=lambda: os.getenv("OPENAI_BASE_URL"))
    temperature: float = 0.0
    max_tokens: int = 2048


@dataclass
class ChunkingConfig:
    """分块配置"""
    # 小块：用于精准检索
    chunk_size: int = 256
    chunk_overlap: int = 50
    # 大块：用于上下文扩展
    parent_chunk_size: int = 1024
    parent_chunk_overlap: int = 100
    # 分块策略
    strategy: str = "sentence_window"  # "fixed", "sentence_window", "semantic"


@dataclass
class RetrievalConfig:
    """检索配置"""
    # 检索数量
    top_k: int = field(default_factory=lambda: int(os.getenv("RETRIEVAL_TOP_K", "10")))
    rerank_top_k: int = field(default_factory=lambda: int(os.getenv("RERANK_TOP_K", "3")))
    # 混合检索权重
    bm25_weight: float = field(default_factory=lambda: float(os.getenv("BM25_WEIGHT", "0.3")))
    dense_weight: float = field(default_factory=lambda: 1 - float(os.getenv("BM25_WEIGHT", "0.3")))
    # 相关性阈值
    relevance_threshold: float = 0.5
    # 是否启用 HyDE
    use_hyde: bool = True
    # 是否启用 Reranker
    use_reranker: bool = True


@dataclass
class VectorStoreConfig:
    """向量数据库配置"""
    persist_directory: str = field(
        default_factory=lambda: os.getenv("CHROMA_PERSIST_DIR", "./data/chroma_db")
    )
    collection_name: str = "advanced_rag"


@dataclass
class Config:
    """总配置"""
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    vector_store: VectorStoreConfig = field(default_factory=VectorStoreConfig)
    
    # 项目路径
    project_root: Path = field(default_factory=lambda: Path(__file__).parent.parent)
    data_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent / "data")


# 全局配置实例
config = Config()
