"""
Advanced RAG 演示脚本

展示完整的 Advanced RAG 流程：
1. 加载示例文档
2. 智能分块和索引
3. 混合检索 + HyDE + Reranker
4. 生成带引用的答案
"""
import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

from src.rag_pipeline import AdvancedRAGPipeline, create_simple_rag, create_advanced_rag
from src.config import config

console = Console()


def demo_simple_rag():
    """演示简单版 RAG（无 HyDE 和 Reranker）"""
    console.print(Panel("[bold]简单 RAG 演示[/bold]", style="blue"))
    
    # 创建简单 RAG
    rag = create_simple_rag()
    
    # 索引示例文档
    docs_dir = Path(__file__).parent.parent / "data" / "sample_docs"
    if docs_dir.exists():
        rag.index_documents(directory_path=str(docs_dir))
    else:
        console.print("[red]示例文档目录不存在，请先创建 data/sample_docs/[/red]")
        return
    
    # 测试问题
    questions = [
        "什么是 RAG？它有什么优势？",
        "HyDE 是什么技术？",
        "LangChain 和 LlamaIndex 有什么区别？",
    ]
    
    for question in questions:
        console.print(f"\n[bold cyan]问题: {question}[/bold cyan]")
        response = rag.query(question)
        rag.print_response(response)
        console.print("\n" + "="*80)


def demo_advanced_rag():
    """演示完整版 Advanced RAG"""
    console.print(Panel("[bold]Advanced RAG 演示[/bold]", style="green"))
    
    # 创建 Advanced RAG
    rag = create_advanced_rag()
    
    # 索引示例文档
    docs_dir = Path(__file__).parent.parent / "data" / "sample_docs"
    if docs_dir.exists():
        rag.index_documents(directory_path=str(docs_dir))
    else:
        console.print("[red]示例文档目录不存在[/red]")
        return
    
    # 测试问题
    questions = [
        "解释一下 Sentence Window Retrieval 的原理",
        "混合检索为什么比单一检索效果好？",
        "如何评估 RAG 系统的效果？",
    ]
    
    for question in questions:
        console.print(f"\n[bold cyan]问题: {question}[/bold cyan]")
        response = rag.query(question)
        rag.print_response(response)
        console.print("\n" + "="*80)


def demo_comparison():
    """对比简单 RAG 和 Advanced RAG 的效果"""
    console.print(Panel("[bold]RAG 效果对比[/bold]", style="magenta"))
    
    # 创建两种 RAG
    simple_rag = create_simple_rag()
    advanced_rag = create_advanced_rag()
    
    # 索引同样的文档
    docs_dir = Path(__file__).parent.parent / "data" / "sample_docs"
    if not docs_dir.exists():
        console.print("[red]示例文档目录不存在[/red]")
        return
    
    console.print("[yellow]索引文档中...[/yellow]")
    simple_rag.index_documents(directory_path=str(docs_dir))
    advanced_rag.index_documents(directory_path=str(docs_dir))
    
    # 测试问题
    question = "RAG 技术的核心组件有哪些？每个组件的作用是什么？"
    
    console.print(f"\n[bold cyan]问题: {question}[/bold cyan]\n")
    
    # 简单 RAG 结果
    console.print("[bold blue]>>> 简单 RAG 结果 <<<[/bold blue]")
    simple_response = simple_rag.query(question)
    simple_rag.print_response(simple_response)
    
    console.print("\n" + "-"*40 + "\n")
    
    # Advanced RAG 结果
    console.print("[bold green]>>> Advanced RAG 结果 <<<[/bold green]")
    advanced_response = advanced_rag.query(question)
    advanced_rag.print_response(advanced_response)


def interactive_mode():
    """交互式问答模式"""
    console.print(Panel("[bold]交互式 RAG 问答[/bold]", style="cyan"))
    
    # 选择 RAG 类型
    console.print("选择 RAG 类型:")
    console.print("1. 简单 RAG（快速，无 HyDE/Reranker）")
    console.print("2. Advanced RAG（完整功能）")
    
    choice = input("\n请选择 (1/2): ").strip()
    
    if choice == "1":
        rag = create_simple_rag()
    else:
        rag = create_advanced_rag()
    
    # 索引文档
    docs_dir = Path(__file__).parent.parent / "data" / "sample_docs"
    if docs_dir.exists():
        rag.index_documents(directory_path=str(docs_dir))
    else:
        console.print("[red]示例文档目录不存在[/red]")
        return
    
    # 交互式问答
    console.print("\n[green]索引完成！输入问题开始问答（输入 'quit' 退出）[/green]\n")
    
    while True:
        question = input("问题: ").strip()
        
        if question.lower() in ["quit", "exit", "q"]:
            console.print("[yellow]再见！[/yellow]")
            break
        
        if not question:
            continue
        
        response = rag.query(question)
        rag.print_response(response)
        console.print()


def main():
    """主函数"""
    console.print(Markdown("""
# Advanced RAG 演示程序

这个演示展示了 LangChain Advanced RAG 的核心功能：
- **智能分块**: Sentence Window Retrieval
- **混合检索**: BM25 + Dense Embedding
- **Query 改写**: HyDE (Hypothetical Document Embeddings)  
- **重排序**: Cross-encoder Reranker

## 运行模式

1. `demo` - 运行简单演示
2. `advanced` - 运行 Advanced RAG 演示
3. `compare` - 对比两种 RAG 的效果
4. `interactive` - 交互式问答

"""))
    
    if len(sys.argv) < 2:
        mode = "demo"
    else:
        mode = sys.argv[1]
    
    if mode == "demo":
        demo_simple_rag()
    elif mode == "advanced":
        demo_advanced_rag()
    elif mode == "compare":
        demo_comparison()
    elif mode == "interactive":
        interactive_mode()
    else:
        console.print(f"[red]未知模式: {mode}[/red]")
        console.print("可用模式: demo, advanced, compare, interactive")


if __name__ == "__main__":
    main()
