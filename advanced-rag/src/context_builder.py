"""Context builder utilities.

Focus:
- Deduplicate near-identical chunks
- Merge adjacent chunks from same source (optional)
- Basic prompt-injection filtering + trust weighting

Keep it simple and deterministic (no LLM calls).

Notes
-----
This is NOT a complete security solution. It provides:
- cheap pattern-based filtering for common prompt injection markers
- a simple source trust score to order evidence

For production, also enforce tool-level permissioning, strict system prompts, and monitoring.
"""

from __future__ import annotations

import re
from typing import Dict, List, Sequence

from langchain_core.documents import Document


_WORD_RE = re.compile(r"[\w\u4e00-\u9fff]+", re.UNICODE)

# Very common prompt-injection / instruction patterns.
# Keep this small and high-precision to avoid false positives.
_INJECTION_PATTERNS: Sequence[re.Pattern[str]] = (
    # English
    re.compile(r"ignore\s+previous\s+instructions", re.I),
    re.compile(r"disregard\s+(all\s+)?(previous|prior)\s+instructions", re.I),
    re.compile(r"system\s+prompt", re.I),
    re.compile(r"developer\s+message", re.I),
    re.compile(r"you\s+are\s+chatgpt", re.I),
    re.compile(r"do\s+not\s+follow\s+the\s+above", re.I),
    # Chinese
    re.compile(r"忽略.*(指令|说明|规则)", re.I),
    re.compile(r"无视.*(指令|说明|规则)", re.I),
    re.compile(r"系统提示词|系统prompt|开发者消息", re.I),
    re.compile(r"你是.*(ChatGPT|助手|AI)", re.I),
)


def _normalize_text(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def _word_set(text: str) -> set[str]:
    return set(_WORD_RE.findall(_normalize_text(text)))


def jaccard_similarity(a: str, b: str) -> float:
    """Cheap similarity for dedup (token set Jaccard)."""
    sa = _word_set(a)
    sb = _word_set(b)
    if not sa or not sb:
        return 0.0
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / union


def dedupe_documents(
    docs: List[Document],
    *,
    threshold: float = 0.9,
) -> List[Document]:
    """Remove near-duplicate documents."""
    kept: List[Document] = []
    for doc in docs:
        if not doc.page_content:
            continue
        is_dup = False
        for k in kept:
            if jaccard_similarity(doc.page_content, k.page_content) >= threshold:
                is_dup = True
                break
        if not is_dup:
            kept.append(doc)
    return kept


def _get_position(doc: Document) -> int | None:
    for key in ("start_index", "start", "offset", "position"):
        val = doc.metadata.get(key)
        if isinstance(val, int):
            return val
    return None


def merge_adjacent_documents(
    docs: List[Document],
    *,
    max_gap: int = 50,
    separator: str = "\n",
) -> List[Document]:
    """Merge docs from same source that are adjacent by position.

    Works only if docs have a stable position in metadata (e.g. start_index).
    If missing, returns docs unchanged.
    """
    if not docs:
        return docs

    if all(_get_position(d) is None for d in docs):
        return docs

    by_source: Dict[str, List[Document]] = {}
    for d in docs:
        src = str(d.metadata.get("source", ""))
        by_source.setdefault(src, []).append(d)

    merged: List[Document] = []
    for _, group in by_source.items():
        group_sorted = sorted(group, key=lambda d: (_get_position(d) is None, _get_position(d) or 0))

        cur = group_sorted[0]
        cur_pos = _get_position(cur)
        for nxt in group_sorted[1:]:
            nxt_pos = _get_position(nxt)
            if cur_pos is not None and nxt_pos is not None and (nxt_pos - cur_pos) <= max_gap:
                cur = Document(
                    page_content=f"{cur.page_content}{separator}{nxt.page_content}",
                    metadata={**cur.metadata},
                )
                cur_pos = nxt_pos
            else:
                merged.append(cur)
                cur = nxt
                cur_pos = nxt_pos
        merged.append(cur)

    return sorted(merged, key=lambda d: (str(d.metadata.get("source", "")), _get_position(d) or 0))


def looks_like_prompt_injection(text: str) -> bool:
    if not text:
        return False
    t = text.strip()
    # Very long single-line instructions are suspicious.
    if len(t) > 2000 and ("ignore" in t.lower() or "忽略" in t):
        return True
    for pat in _INJECTION_PATTERNS:
        if pat.search(t):
            return True
    return False


def filter_prompt_injection(docs: List[Document]) -> List[Document]:
    """Drop docs that look like prompt injection."""
    return [d for d in docs if not looks_like_prompt_injection(d.page_content)]


def source_trust_score(doc: Document) -> float:
    """Heuristic trust score from metadata.source.

    Higher means more trusted.
    """
    src = str(doc.metadata.get("source", "")).lower()
    if not src:
        return 0.5
    if src.startswith("/") or src.endswith(".md") or src.endswith(".txt"):
        # local knowledge base file
        return 0.9
    if src.startswith("https://"):
        return 0.7
    if src.startswith("http://"):
        return 0.6
    return 0.5


def sort_by_trust_then_length(docs: List[Document]) -> List[Document]:
    """Sort evidence by trust desc, then shorter first (less token noise)."""
    return sorted(
        docs,
        key=lambda d: (-source_trust_score(d), len(d.page_content or "")),
    )
