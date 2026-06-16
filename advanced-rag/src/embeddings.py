"""
Embedding 模块

支持：
1. OpenAI Embeddings (text-embedding-3-small/large)
2. 本地 HuggingFace 模型 (BGE, M3E 等)
"""
from typing import List, Optional
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings

from .config import config


def get_embeddings(
    model_name: Optional[str] = None,
    use_local: Optional[bool] = None,
) -> Embeddings:
    """
    获取 Embedding 模型
    
    Args:
        model_name: 模型名称，不指定则使用配置
        use_local: 是否使用本地模型
    
    Returns:
        Embeddings 实例
    """
    model_name = model_name or config.embedding.model_name
    use_local = use_local if use_local is not None else config.embedding.use_local
    
    if use_local:
        return get_local_embeddings(model_name)
    else:
        return get_openai_embeddings(model_name)


def get_openai_embeddings(model_name: str = "text-embedding-3-small") -> Embeddings:
    """获取 OpenAI Embedding"""
    return OpenAIEmbeddings(
        model=model_name,
        openai_api_key=config.llm.api_key,
        openai_api_base=config.llm.base_url,
    )


def get_local_embeddings(model_name: str = "BAAI/bge-small-zh-v1.5") -> Embeddings:
    """
    获取本地 HuggingFace Embedding
    
    推荐模型：
    - BAAI/bge-small-zh-v1.5 (中文，体积小)
    - BAAI/bge-large-zh-v1.5 (中文，效果好)
    - sentence-transformers/all-MiniLM-L6-v2 (英文，快速)
    """
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        
        return HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cpu"},  # 或 "cuda"
            encode_kwargs={"normalize_embeddings": True},  # BGE 模型建议归一化
        )
    except ImportError:
        raise ImportError(
            "请安装 sentence-transformers: pip install sentence-transformers"
        )


class CachedEmbeddings(Embeddings):
    """
    带缓存的 Embedding 封装
    
    避免重复计算相同文本的 embedding
    """
    
    def __init__(self, base_embeddings: Embeddings):
        self.base_embeddings = base_embeddings
        self._cache = {}
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """批量计算文档 embedding，带缓存"""
        results = []
        texts_to_embed = []
        indices_to_embed = []
        
        for i, text in enumerate(texts):
            if text in self._cache:
                results.append(self._cache[text])
            else:
                texts_to_embed.append(text)
                indices_to_embed.append(i)
                results.append(None)  # 占位
        
        # 批量计算未缓存的
        if texts_to_embed:
            new_embeddings = self.base_embeddings.embed_documents(texts_to_embed)
            for idx, (text, emb) in enumerate(zip(texts_to_embed, new_embeddings)):
                self._cache[text] = emb
                results[indices_to_embed[idx]] = emb
        
        return results
    
    def embed_query(self, text: str) -> List[float]:
        """计算查询 embedding，带缓存"""
        if text not in self._cache:
            self._cache[text] = self.base_embeddings.embed_query(text)
        return self._cache[text]
    
    def clear_cache(self):
        """清空缓存"""
        self._cache = {}


if __name__ == "__main__":
    # 测试
    print("测试 Embedding 模块...")
    
    # 使用本地模型测试（不需要 API Key）
    try:
        embeddings = get_local_embeddings()
        
        test_texts = [
            "什么是 RAG？",
            "RAG 是检索增强生成技术",
            "今天天气真好"
        ]
        
        vectors = embeddings.embed_documents(test_texts)
        
        print(f"生成了 {len(vectors)} 个向量")
        print(f"向量维度: {len(vectors[0])}")
        
        # 计算相似度
        import numpy as np
        
        def cosine_sim(a, b):
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        
        print(f"\n相似度测试:")
        print(f"'{test_texts[0]}' vs '{test_texts[1]}': {cosine_sim(vectors[0], vectors[1]):.4f}")
        print(f"'{test_texts[0]}' vs '{test_texts[2]}': {cosine_sim(vectors[0], vectors[2]):.4f}")
        
    except Exception as e:
        print(f"本地模型测试失败: {e}")
        print("尝试使用 OpenAI Embedding（需要配置 API Key）")
