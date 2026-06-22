"""
RLM Doc Agent - 递归文档理解 Agent
====================================

核心思路：
  LLM 通过调用工具来"感知"文档结构，然后自主决定如何递归拆解。
  整个流程是 LLM 驱动的 Tool Use 循环，不是硬编码逻辑。

工具集（LLM 可以调用）：
  - list_documents()           → 列出可用文档
  - extract_toc(path)          → 提取文档目录/章节结构
  - read_section(path, title)  → 读取指定章节的内容
  - read_chunk(path, start, end) → 按字符位置读取片段（fallback）
  - summarize_text(text, question) → 对一段文本做针对性摘要
  - merge_summaries(summaries, question) → 合并多段摘要得出最终答案

执行流程（Agent 自主决策）：
  1. LLM 调用 list_documents() 了解有哪些文档
  2. LLM 调用 extract_toc() 获取文档结构
  3. LLM 根据 TOC 决定拆解策略（按章节/按二级标题/按 chunk）
  4. LLM 对每个部分调用 read_section() + summarize_text()
  5. LLM 调用 merge_summaries() 汇总最终答案
  6. 如果某个章节太长，LLM 会再次递归拆解

泛化设计：
  - 支持 Markdown (.md) 文档（通过 # 标题解析 TOC）
  - 支持纯文本（fallback 到 chunk 切割）
  - 可扩展支持 PDF（替换 extract_toc 实现即可）
  - 工具接口标准化，LLM 只需知道工具名和参数
"""

import os
import re
import json
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(dotenv_path="../advanced-rag/.env")

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
)
MODEL = os.getenv("LLM_MODEL")

DOCS_DIR = Path(__file__).parent / "sample_docs"
MAX_SECTION_CHARS = 3000  # 超过这个长度的章节需要再次拆解

# ─────────────────────────────────────────────────────────
# 工具实现（Python 函数）
# LLM 看不到实现，只看到工具描述和参数
# ─────────────────────────────────────────────────────────

def list_documents() -> dict:
    """列出 sample_docs 目录下所有可用文档"""
    docs = []
    for f in DOCS_DIR.iterdir():
        if f.suffix in (".md", ".txt"):
            size = f.stat().st_size
            docs.append({
                "filename": f.name,
                "path": str(f),
                "size_chars": size,
                "type": "markdown" if f.suffix == ".md" else "text"
            })
    return {"documents": docs}


def extract_toc(path: str) -> dict:
    """
    提取文档的目录结构。
    对 Markdown：解析 # 标题层级
    对纯文本：按段落估算分块
    """
    p = Path(path)
    if not p.exists():
        return {"error": f"文件不存在: {path}"}

    text = p.read_text(encoding="utf-8")

    if p.suffix == ".md":
        sections = []
        lines = text.split("\n")
        char_pos = 0

        for i, line in enumerate(lines):
            match = re.match(r"^(#{1,4})\s+(.+)", line)
            if match:
                level = len(match.group(1))
                title = match.group(2).strip()
                sections.append({
                    "level": level,
                    "title": title,
                    "line_num": i + 1,
                    "char_start": char_pos,
                })
            char_pos += len(line) + 1  # +1 for \n

        # 计算每个章节的结束位置和字符长度
        for i, sec in enumerate(sections):
            next_start = sections[i + 1]["char_start"] if i + 1 < len(sections) else len(text)
            sec["char_end"] = next_start
            sec["char_length"] = next_start - sec["char_start"]

        return {
            "total_chars": len(text),
            "total_sections": len(sections),
            "sections": sections
        }
    else:
        # 纯文本：按段落切分
        paragraphs = [p for p in text.split("\n\n") if p.strip()]
        chunk_size = MAX_SECTION_CHARS
        chunks = []
        pos = 0
        for i in range(0, len(text), chunk_size):
            chunks.append({
                "level": 1,
                "title": f"片段 {len(chunks)+1}",
                "char_start": i,
                "char_end": min(i + chunk_size, len(text)),
                "char_length": min(chunk_size, len(text) - i)
            })
        return {
            "total_chars": len(text),
            "total_sections": len(chunks),
            "sections": chunks
        }


def read_section(path: str, title: str) -> dict:
    """读取文档中指定标题的章节内容"""
    toc = extract_toc(path)
    if "error" in toc:
        return toc

    text = Path(path).read_text(encoding="utf-8")

    for sec in toc["sections"]:
        if sec["title"] == title:
            content = text[sec["char_start"]:sec["char_end"]].strip()
            return {
                "title": title,
                "content": content,
                "char_length": sec["char_length"],
                "needs_further_split": sec["char_length"] > MAX_SECTION_CHARS
            }

    return {"error": f"未找到章节: {title}"}


def read_chunk(path: str, char_start: int, char_end: int) -> dict:
    """按字符位置读取文档片段（当章节过大时的 fallback）"""
    p = Path(path)
    if not p.exists():
        return {"error": f"文件不存在: {path}"}

    text = p.read_text(encoding="utf-8")
    content = text[char_start:char_end].strip()
    return {
        "content": content,
        "char_start": char_start,
        "char_end": char_end,
        "char_length": char_end - char_start
    }


def summarize_text(text: str, question: str) -> dict:
    """对一段文本做针对用户问题的摘要（内部 LLM 调用）"""
    prompt = f"""请针对以下问题，摘要这段文本的关键内容。

用户问题：{question}

文本内容：
{text[:4000]}  

要求：
- 只保留与问题相关的信息
- 摘要控制在 200 字以内
- 如果这段文本与问题无关，回复"[与问题无关]"
"""
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    summary = response.choices[0].message.content.strip()
    return {"summary": summary}


def merge_summaries(summaries: list[dict], question: str) -> dict:
    """合并多段摘要，得出针对用户问题的最终答案"""
    # 过滤掉无关的摘要
    relevant = [s for s in summaries if "[与问题无关]" not in s.get("summary", "")]

    if not relevant:
        return {"answer": "在文档中未找到与该问题相关的内容。"}

    combined = "\n\n".join([
        f"【{s.get('section', '片段')}】\n{s['summary']}"
        for s in relevant
    ])

    prompt = f"""基于以下各章节摘要，综合回答用户的问题。

用户问题：{question}

各章节摘要：
{combined}

请给出完整、清晰的最终答案："""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    answer = response.choices[0].message.content.strip()
    return {"answer": answer}


# ─────────────────────────────────────────────────────────
# 工具注册表（告诉 LLM 有哪些工具可以用）
# 遵循 OpenAI Function Calling 格式
# ─────────────────────────────────────────────────────────

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "list_documents",
            "description": "列出所有可用的文档文件，返回文件名、路径、大小等信息。在开始处理任务前先调用此工具了解有哪些文档。",
            "parameters": {"type": "object", "properties": {}, "required": []}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "extract_toc",
            "description": "提取文档的目录结构（Table of Contents）。返回各章节标题、层级、字符位置和长度。这是了解文档结构的核心工具，在读取内容前必须先调用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "文档的完整路径"}
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_section",
            "description": "按标题读取文档的某个章节内容。返回内容和 needs_further_split 标志（如果章节太长，需要进一步拆分）。",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "文档的完整路径"},
                    "title": {"type": "string", "description": "章节标题，必须与 extract_toc 返回的 title 完全一致"}
                },
                "required": ["path", "title"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_chunk",
            "description": "按字符位置范围读取文档片段。当章节太长需要进一步切割时使用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "文档的完整路径"},
                    "char_start": {"type": "integer", "description": "起始字符位置"},
                    "char_end": {"type": "integer", "description": "结束字符位置"}
                },
                "required": ["path", "char_start", "char_end"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "summarize_text",
            "description": "对一段文本做针对用户问题的摘要。每次读取到章节内容后调用此工具提炼要点。",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "需要摘要的文本内容"},
                    "question": {"type": "string", "description": "用户的原始问题，摘要要围绕这个问题"}
                },
                "required": ["text", "question"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "merge_summaries",
            "description": "将所有章节的摘要合并，生成针对用户问题的最终完整答案。在处理完所有相关章节后调用此工具。",
            "parameters": {
                "type": "object",
                "properties": {
                    "summaries": {
                        "type": "array",
                        "description": "章节摘要列表，每项包含 section（章节名）和 summary（摘要内容）",
                        "items": {
                            "type": "object",
                            "properties": {
                                "section": {"type": "string"},
                                "summary": {"type": "string"}
                            }
                        }
                    },
                    "question": {"type": "string", "description": "用户的原始问题"}
                },
                "required": ["summaries", "question"]
            }
        }
    }
]

# ─────────────────────────────────────────────────────────
# 工具分发器（把 LLM 的工具调用请求路由到对应 Python 函数）
# ─────────────────────────────────────────────────────────

TOOL_FUNCTIONS = {
    "list_documents": lambda args: list_documents(),
    "extract_toc": lambda args: extract_toc(args["path"]),
    "read_section": lambda args: read_section(args["path"], args["title"]),
    "read_chunk": lambda args: read_chunk(args["path"], args["char_start"], args["char_end"]),
    "summarize_text": lambda args: summarize_text(args["text"], args["question"]),
    "merge_summaries": lambda args: merge_summaries(args["summaries"], args["question"]),
}


def execute_tool(tool_name: str, tool_args: dict) -> str:
    """执行工具调用，返回 JSON 字符串结果"""
    if tool_name not in TOOL_FUNCTIONS:
        return json.dumps({"error": f"未知工具: {tool_name}"}, ensure_ascii=False)
    result = TOOL_FUNCTIONS[tool_name](tool_args)
    return json.dumps(result, ensure_ascii=False, indent=2)


# ─────────────────────────────────────────────────────────
# Agent 主循环
# LLM 在这里自主决策：调用哪个工具、用什么参数、何时结束
# ─────────────────────────────────────────────────────────

SYSTEM_PROMPT = """你是一个专门处理大型文档的递归理解 Agent。

你的任务是：通过调用工具来理解用户给定的文档，然后回答用户的问题。

工作策略（必须遵循）：
1. 先调用 list_documents() 了解有哪些文档
2. 对目标文档调用 extract_toc() 获取结构
3. 根据 TOC 判断哪些章节与问题相关
4. 对每个相关章节调用 read_section()，然后立即调用 summarize_text() 提炼要点
5. 如果某章节的 needs_further_split=true，则用 read_chunk() 分段处理
6. 收集完所有章节摘要后，调用 merge_summaries() 给出最终答案

重要原则：
- 不要一次性读取整个文档（太长）
- 每读一段，立刻摘要，不要积累大量原文
- 优先处理标题与问题直接相关的章节
- 工具调用结果可能包含 needs_further_split=true，这时要主动拆分
"""


def run_agent(question: str, verbose: bool = True) -> str:
    """
    运行 RLM Doc Agent。

    Args:
        question: 用户问题
        verbose: 是否打印详细的工具调用过程

    Returns:
        最终答案
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question}
    ]

    step = 0
    max_steps = 30  # 防止无限循环

    print(f"\n{'='*60}")
    print(f"🎯 用户问题: {question}")
    print(f"{'='*60}")

    while step < max_steps:
        step += 1

        # ── LLM 决策：下一步调用哪个工具 ──
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            temperature=0,
        )

        msg = response.choices[0].message
        finish_reason = response.choices[0].finish_reason

        # ── 终止条件：LLM 决定不再调用工具，直接输出答案 ──
        if finish_reason == "stop":
            final_answer = msg.content
            print(f"\n{'='*60}")
            print("✅ Agent 完成，最终答案：")
            print(f"{'='*60}")
            print(final_answer)
            return final_answer

        # ── 处理工具调用 ──
        if not msg.tool_calls:
            # 没有工具调用也没有 stop，异常情况
            break

        # 把 LLM 的工具调用决策加入消息历史
        messages.append({"role": "assistant", "content": msg.content, "tool_calls": [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.function.name, "arguments": tc.function.arguments}
            }
            for tc in msg.tool_calls
        ]})

        # ── 逐个执行工具调用 ──
        for tool_call in msg.tool_calls:
            tool_name = tool_call.function.name
            try:
                tool_args = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError:
                tool_args = {}

            if verbose:
                # 打印工具调用（隐藏大段文本参数）
                display_args = {
                    k: (v[:80] + "..." if isinstance(v, str) and len(v) > 80 else v)
                    for k, v in tool_args.items()
                }
                print(f"\n[Step {step}] 🔧 调用工具: {tool_name}")
                print(f"          参数: {json.dumps(display_args, ensure_ascii=False)}")

            # 执行工具
            result_str = execute_tool(tool_name, tool_args)
            result_data = json.loads(result_str)

            if verbose:
                # 打印结果摘要
                if tool_name == "extract_toc":
                    sections = result_data.get("sections", [])
                    print(f"          结果: 找到 {len(sections)} 个章节")
                    for s in sections[:8]:  # 只打印前8个
                        indent = "  " * (s["level"] - 1)
                        print(f"                {indent}{'#'*s['level']} {s['title']} ({s['char_length']} 字)")
                    if len(sections) > 8:
                        print(f"                ... 还有 {len(sections)-8} 个章节")
                elif tool_name == "summarize_text":
                    summary = result_data.get("summary", "")
                    print(f"          摘要: {summary[:120]}...")
                elif tool_name == "merge_summaries":
                    answer = result_data.get("answer", "")
                    print(f"          合并完成，答案长度: {len(answer)} 字")
                elif tool_name == "list_documents":
                    docs = result_data.get("documents", [])
                    print(f"          结果: 找到 {len(docs)} 个文档")
                    for d in docs:
                        print(f"                - {d['filename']} ({d['size_chars']} 字节)")
                else:
                    # 通用：打印结果前100字
                    result_preview = result_str[:150].replace("\n", " ")
                    print(f"          结果: {result_preview}...")

            # 把工具执行结果反馈给 LLM
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result_str
            })

    print(f"\n⚠️ 达到最大步骤数 ({max_steps})，Agent 终止。")
    return "处理超时"


# ─────────────────────────────────────────────────────────
# 主入口
# ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("🔄 RLM Doc Agent - 递归文档理解 + 工具调用")
    print("文档目录:", DOCS_DIR)

    # ── 示例问题列表（可以换任意问题）──
    questions = [
        "这篇论文的核心创新是什么？Transformer 相比 RNN/CNN 的主要优势在哪里？",
        # "作者是如何处理位置信息的？正弦位置编码的公式是什么？",
        # "在机器翻译任务上，Transformer 取得了什么成绩？",
    ]

    for q in questions:
        answer = run_agent(q, verbose=True)
        print("\n" + "─" * 60 + "\n")
