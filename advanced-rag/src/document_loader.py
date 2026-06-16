"""
文档加载和智能分块模块

实现了多种分块策略：
1. Fixed Chunking - 固定大小分块
2. Sentence Window - 句子窗口（小块检索，大块返回）
3. Semantic Chunking - 语义分块（基于相似度边界）
"""
import re
from typing import List, Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
)
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    SentenceTransformersTokenTextSplitter,
)

from .config import config


@dataclass
class ChunkedDocument:
    """分块后的文档，包含小块和对应的父块"""
    chunk: Document           # 小块（用于检索）
    parent_chunk: Document    # 父块（用于上下文）
    chunk_index: int
    parent_index: int


class DocumentLoader:
    """文档加载器，支持多种格式"""
    
    SUPPORTED_EXTENSIONS = {
        ".pdf": PyPDFLoader,
        ".txt": TextLoader,
        ".md": TextLoader,  # 使用 TextLoader 加载 markdown，避免依赖问题
    }
    
    def __init__(self):
        pass
    
    def load(self, file_path: str | Path) -> List[Document]:
        """加载单个文档"""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        extension = file_path.suffix.lower()
        
        if extension not in self.SUPPORTED_EXTENSIONS:
            # 尝试作为文本文件加载
            loader = TextLoader(str(file_path), encoding="utf-8")
        else:
            loader_class = self.SUPPORTED_EXTENSIONS[extension]
            if extension == ".txt":
                loader = loader_class(str(file_path), encoding="utf-8")
            else:
                loader = loader_class(str(file_path))
        
        docs = loader.load()
        
        # 添加元数据
        for doc in docs:
            doc.metadata["source"] = str(file_path)
            doc.metadata["filename"] = file_path.name
        
        return docs
    
    def load_directory(self, dir_path: str | Path) -> List[Document]:
        """加载目录下的所有文档"""
        dir_path = Path(dir_path)
        all_docs = []
        
        for ext in self.SUPPORTED_EXTENSIONS.keys():
            for file_path in dir_path.rglob(f"*{ext}"):
                try:
                    docs = self.load(file_path)
                    all_docs.extend(docs)
                except Exception as e:
                    print(f"加载文件失败 {file_path}: {e}")
        
        return all_docs


class SmartChunker:
    """
    智能分块器
    
    实现 Sentence Window Retrieval 策略：
    - 创建小块用于精准检索
    - 保留小块到大块的映射，检索时返回更完整的上下文
    """
    
    def __init__(
        self,
        chunk_size: int = None,
        chunk_overlap: int = None,
        parent_chunk_size: int = None,
        parent_chunk_overlap: int = None,
    ):
        self.chunk_size = chunk_size or config.chunking.chunk_size
        self.chunk_overlap = chunk_overlap or config.chunking.chunk_overlap
        self.parent_chunk_size = parent_chunk_size or config.chunking.parent_chunk_size
        self.parent_chunk_overlap = parent_chunk_overlap or config.chunking.parent_chunk_overlap
        
        # 小块分割器
        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?", " ", ""],
            length_function=len,
        )
        
        # 大块分割器
        self.parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.parent_chunk_size,
            chunk_overlap=self.parent_chunk_overlap,
            separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?", " ", ""],
            length_function=len,
        )
    
    def chunk_documents(
        self, 
        documents: List[Document]
    ) -> tuple[List[Document], Dict[str, Document]]:
        """
        对文档进行分块
        
        Returns:
            child_chunks: 小块列表（用于构建索引）
            parent_mapping: chunk_id -> parent_chunk 的映射
        """
        child_chunks = []
        parent_mapping = {}
        
        for doc in documents:
            # 先分大块
            parent_chunks = self.parent_splitter.split_documents([doc])
            
            for parent_idx, parent_chunk in enumerate(parent_chunks):
                # 为每个大块生成唯一ID
                parent_id = f"{doc.metadata.get('source', 'unknown')}_{parent_idx}"
                parent_chunk.metadata["parent_id"] = parent_id
                parent_chunk.metadata["parent_index"] = parent_idx
                
                # 再分小块
                small_chunks = self.child_splitter.split_documents([parent_chunk])
                
                for child_idx, child_chunk in enumerate(small_chunks):
                    # 小块保留对父块的引用
                    chunk_id = f"{parent_id}_child_{child_idx}"
                    child_chunk.metadata["chunk_id"] = chunk_id
                    child_chunk.metadata["parent_id"] = parent_id
                    child_chunk.metadata["child_index"] = child_idx
                    
                    child_chunks.append(child_chunk)
                    parent_mapping[chunk_id] = parent_chunk
        
        return child_chunks, parent_mapping
    
    def simple_chunk(self, documents: List[Document]) -> List[Document]:
        """简单分块（不使用 Sentence Window）"""
        return self.child_splitter.split_documents(documents)


class SemanticChunker:
    """
    语义分块器（进阶）
    
    基于语义相似度在边界处切分，而不是固定长度。
    当连续句子的相似度低于阈值时进行切分。
    """
    
    def __init__(self, embeddings, threshold: float = 0.5):
        self.embeddings = embeddings
        self.threshold = threshold
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """将文本分割成句子"""
        # 支持中英文句子
        pattern = r'(?<=[。！？.!?])\s*'
        sentences = re.split(pattern, text)
        return [s.strip() for s in sentences if s.strip()]
    
    def chunk_document(self, document: Document) -> List[Document]:
        """基于语义边界分块"""
        sentences = self._split_into_sentences(document.page_content)
        
        if len(sentences) <= 1:
            return [document]
        
        # 计算每个句子的 embedding
        embeddings = self.embeddings.embed_documents(sentences)
        
        # 计算相邻句子的相似度
        chunks = []
        current_chunk_sentences = [sentences[0]]
        
        for i in range(1, len(sentences)):
            # 计算与前一个句子的余弦相似度
            similarity = self._cosine_similarity(embeddings[i-1], embeddings[i])
            
            if similarity < self.threshold:
                # 相似度低于阈值，开始新块
                chunk_text = " ".join(current_chunk_sentences)
                chunks.append(Document(
                    page_content=chunk_text,
                    metadata={**document.metadata, "chunk_method": "semantic"}
                ))
                current_chunk_sentences = [sentences[i]]
            else:
                current_chunk_sentences.append(sentences[i])
        
        # 添加最后一块
        if current_chunk_sentences:
            chunk_text = " ".join(current_chunk_sentences)
            chunks.append(Document(
                page_content=chunk_text,
                metadata={**document.metadata, "chunk_method": "semantic"}
            ))
        
        return chunks
    
    @staticmethod
    def _cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
        """计算余弦相似度"""
        import numpy as np
        vec1, vec2 = np.array(vec1), np.array(vec2)
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


if __name__ == "__main__":
    # 测试代码
    loader = DocumentLoader()
    chunker = SmartChunker()
    
    # 创建测试文档
    test_doc = Document(
        page_content="""
        RAG（Retrieval-Augmented Generation）是一种结合检索和生成的技术。
        
        它的核心思想是：在生成答案之前，先从知识库中检索相关文档，
        然后将检索到的内容作为上下文，帮助大语言模型生成更准确的回答。
        
        RAG 的主要优势包括：
        1. 减少幻觉：通过提供真实的参考资料
        2. 知识更新：无需重新训练模型即可更新知识
        3. 可解释性：可以追溯答案的来源
        
        工业级 RAG 通常包含以下组件：
        - 文档加载和预处理
        - 智能分块策略
        - 多路召回（稀疏+稠密）
        - 重排序和过滤
        - 上下文压缩
        """,
        metadata={"source": "test.txt"}
    )
    
    child_chunks, parent_mapping = chunker.chunk_documents([test_doc])
    
    print(f"生成了 {len(child_chunks)} 个小块")
    print(f"父块映射数量: {len(parent_mapping)}")
    
    for i, chunk in enumerate(child_chunks[:3]):
        print(f"\n--- Chunk {i} ---")
        print(f"内容: {chunk.page_content[:100]}...")
        print(f"元数据: {chunk.metadata}")
