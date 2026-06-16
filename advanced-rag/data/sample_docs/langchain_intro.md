# LangChain 框架介绍

## 什么是 LangChain？

LangChain 是一个用于构建大语言模型（LLM）应用的开源框架。它提供了一套工具和抽象，帮助开发者快速构建基于 LLM 的应用程序。

## 核心概念

### 1. Components（组件）

LangChain 提供了多种可重用的组件：

- **LLMs/Chat Models**：对接各种大语言模型
- **Prompts**：提示词模板管理
- **Embeddings**：文本向量化
- **Vector Stores**：向量数据库
- **Document Loaders**：文档加载器
- **Text Splitters**：文本分割器
- **Retrievers**：检索器

### 2. Chains（链）

Chain 是将多个组件串联起来的工作流：

```python
chain = prompt | llm | output_parser
result = chain.invoke({"input": "Hello"})
```

### 3. Agents（代理）

Agent 可以根据用户输入动态决定调用哪些工具：

- 理解用户意图
- 选择合适的工具
- 执行并返回结果
- 支持多轮交互

### 4. LCEL（LangChain Expression Language）

LangChain 的声明式语法，用管道符 `|` 连接组件：

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_template("告诉我关于 {topic} 的知识")
llm = ChatOpenAI()
output_parser = StrOutputParser()

chain = prompt | llm | output_parser
```

## LangChain 构建 RAG

### 基本流程

```python
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

# 1. 加载文档
loader = PyPDFLoader("document.pdf")
docs = loader.load()

# 2. 分块
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(docs)

# 3. 向量化并存储
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(chunks, embeddings)

# 4. 创建检索器
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# 5. 创建 RAG Chain
prompt = ChatPromptTemplate.from_template("""
基于以下上下文回答问题：
{context}

问题：{question}
""")

llm = ChatOpenAI()

rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
)

# 6. 执行查询
answer = rag_chain.invoke("什么是机器学习？")
```

## LangChain vs LlamaIndex

| 特性 | LangChain | LlamaIndex |
|-----|-----------|------------|
| 定位 | 通用 LLM 应用框架 | 专注于数据索引和检索 |
| 灵活性 | 高，组件可自由组合 | 中等，更多开箱即用 |
| RAG 支持 | 需要自己组装 | 内置多种 RAG 策略 |
| Agent | 功能强大 | 相对简单 |
| 学习曲线 | 较陡 | 较平缓 |

## 最佳实践

1. **使用 LCEL**：声明式语法更清晰
2. **异步优先**：使用 `ainvoke` 提升性能
3. **流式输出**：使用 `stream` 提升用户体验
4. **错误处理**：添加 `with_fallbacks` 处理失败
5. **缓存**：使用 `CacheBackedEmbeddings` 减少重复计算
