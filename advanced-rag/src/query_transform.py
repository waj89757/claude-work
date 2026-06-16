"""
Query 转换模块

实现多种 Query 改写策略：
1. HyDE (Hypothetical Document Embeddings) - 生成假设答案再检索
2. Query Expansion - 查询扩展
3. Multi-Query - 多角度查询
"""
from typing import List, Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

from .config import config


class QueryTransformer:
    """
    Query 转换器
    
    支持多种转换策略来提升检索效果
    """
    
    def __init__(self, llm: Optional[ChatOpenAI] = None):
        self.llm = llm or ChatOpenAI(
            model=config.llm.model_name,
            temperature=0,
            api_key=config.llm.api_key,
            base_url=config.llm.base_url,
        )
    
    def hyde(self, query: str) -> str:
        """
        HyDE (Hypothetical Document Embeddings)
        
        思路：用 LLM 先生成一个假设性答案，然后用这个答案的 embedding 去检索。
        因为答案的语义和真实文档更接近，所以检索效果更好。
        
        论文：https://arxiv.org/abs/2212.10496
        """
        prompt = ChatPromptTemplate.from_template("""请根据以下问题，写一段可能的答案。
这个答案不需要是正确的，只需要像是一段真实的文档内容。

问题: {query}

假设性答案:""")
        
        chain = prompt | self.llm | StrOutputParser()
        hypothetical_answer = chain.invoke({"query": query})
        
        return hypothetical_answer
    
    def expand_query(self, query: str) -> str:
        """
        Query 扩展
        
        添加同义词、相关词来扩展查询，提高召回率
        """
        prompt = ChatPromptTemplate.from_template("""请扩展以下查询，添加相关的同义词、近义词和关键术语。
保持原始查询的含义，但添加更多相关词汇以帮助检索。

原始查询: {query}

扩展后的查询（保持简洁，不超过50字）:""")
        
        chain = prompt | self.llm | StrOutputParser()
        expanded_query = chain.invoke({"query": query})
        
        return expanded_query
    
    def multi_query(self, query: str, num_queries: int = 3) -> List[str]:
        """
        Multi-Query 转换
        
        从不同角度生成多个查询，然后合并检索结果
        """
        prompt = ChatPromptTemplate.from_template("""请从不同角度重写以下问题，生成 {num_queries} 个不同的查询。
每个查询应该表达相似的意图，但使用不同的措辞或关注不同的方面。

原始问题: {query}

请用换行分隔每个查询，不要编号:""")
        
        chain = prompt | self.llm | StrOutputParser()
        result = chain.invoke({"query": query, "num_queries": num_queries})
        
        queries = [q.strip() for q in result.strip().split("\n") if q.strip()]
        return queries[:num_queries]
    
    def step_back(self, query: str) -> str:
        """
        Step-Back Prompting
        
        先问一个更宽泛的问题，获取背景知识后再回答具体问题
        论文：https://arxiv.org/abs/2310.06117
        """
        prompt = ChatPromptTemplate.from_template("""给定一个具体的问题，请生成一个更抽象、更宽泛的"后退问题"。
这个后退问题应该帮助我们获取回答原问题所需的背景知识。

具体问题: {query}

后退问题:""")
        
        chain = prompt | self.llm | StrOutputParser()
        step_back_query = chain.invoke({"query": query})
        
        return step_back_query
    
    # ------------------------------------------------------------------
    # 判断是否是复杂问题：先用规则快速筛（0 cost），再用 LLM 兜底
    # ------------------------------------------------------------------

    _COMPLEX_SIGNALS = [
        # 比较/对比意图
        "区别", "对比", "比较", "vs", "和.*有什么不同",
        # 多个实体或概念并列
        "以及", "和.*关系", "如何.*同时",
        # 因果+条件链
        "为什么.*才能", "如果.*怎么",
        # 多步骤意图
        "步骤", "流程", "怎么.*然后",
        # 英文
        "difference between", "compare", "relationship between",
        "how does.*affect",
    ]

    def is_complex_query(self, query: str) -> bool:
        """
        判断问题是否复杂，需要多跳检索。

        策略（按成本从低到高）：
        1. 规则快筛：命中关键词 → 复杂
        2. 长度：超过 40 字且含多个问号 → 复杂
        3. 以上都不命中 → 简单（直接做单次检索，不调 LLM，节省成本）

        注：工业上还可以用一个小分类模型（BERT tiny 级别）替换这里的规则，
        准确率更高，但成本和这里差不多。
        """
        import re
        q = query.strip()

        # 规则 1：信号词命中
        for signal in self._COMPLEX_SIGNALS:
            if re.search(signal, q, re.I):
                return True

        # 规则 2：问题很长且含多个问号/逗号，大概率是复合问题
        if len(q) > 40 and (q.count("？") + q.count("?") >= 2 or q.count("，") >= 3):
            return True

        return False

    def decompose(self, query: str) -> List[str]:
        """
        Query 分解（多跳检索的第一步）

        流程：
        1. is_complex_query() 判断是否需要分解（不需要就直接返回 [query]）
        2. 需要：让 LLM 拆成 2-4 个子问题，每个子问题可以独立检索
        3. 分解后每个子问题单独检索，检索结果合并送给 LLM

        例子：
        原问题："BGE 和 OpenAI Embedding 的区别，以及哪个更适合中文 RAG？"
        子问题1："BGE 是什么，有哪些特点？"
        子问题2："OpenAI Embedding 是什么，有哪些特点？"
        子问题3："中文 RAG 的 Embedding 模型选型标准是什么？"
        → 三路检索 → 合并 context → LLM 综合回答
        """
        # 快速路径：简单问题不分解
        if not self.is_complex_query(query):
            return [query]

        prompt = ChatPromptTemplate.from_template("""请将以下复杂问题分解为2-4个更简单的子问题。
要求：
- 每个子问题可以独立检索，单独回答
- 子问题组合起来能完整回答原问题
- 子问题不要重复，不要废话

复杂问题: {query}

子问题（每行一个，不要编号）:""")

        chain = prompt | self.llm | StrOutputParser()
        result = chain.invoke({"query": query})

        sub_queries = [q.strip() for q in result.strip().split("\n") if q.strip()]
        # 去掉 LLM 可能加的编号：1. 2. (1) 等
        sub_queries = [q.lstrip("0123456789.-) ") for q in sub_queries]
        # 过滤空字符串
        sub_queries = [q for q in sub_queries if len(q) > 3]

        # 兜底：万一 LLM 没拆出来，就用原问题
        return sub_queries if sub_queries else [query]


if __name__ == "__main__":
    # 测试 Query 转换
    transformer = QueryTransformer()
    
    test_query = "什么是 RAG？它和 Fine-tuning 有什么区别？"
    
    print("=== HyDE ===")
    hyde_result = transformer.hyde(test_query)
    print(f"假设答案: {hyde_result}\n")
    
    print("=== Query 扩展 ===")
    expanded = transformer.expand_query(test_query)
    print(f"扩展后: {expanded}\n")
    
    print("=== Multi-Query ===")
    multi = transformer.multi_query(test_query)
    for i, q in enumerate(multi, 1):
        print(f"{i}. {q}")
    print()
    
    print("=== Step-Back ===")
    step_back = transformer.step_back(test_query)
    print(f"后退问题: {step_back}\n")
    
    print("=== Query 分解 ===")
    sub_queries = transformer.decompose(test_query)
    for i, q in enumerate(sub_queries, 1):
        print(f"{i}. {q}")
