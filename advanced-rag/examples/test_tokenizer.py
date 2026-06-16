"""测试自定义词典效果"""
import jieba
from pathlib import Path

# 测试词
test_words = [
    "RAG技术栈",
    "LangChain框架",
    "向量数据库",
    "检索增强生成技术",
]

print("=== 未加载词典 ===")
for word in test_words:
    tokens = list(jieba.cut(word))
    print(f"{word} → {tokens}")

print("\n=== 加载词典后 ===")
dict_path = Path(__file__).parent.parent / "data" / "custom_dict.txt"
jieba.load_userdict(str(dict_path))

for word in test_words:
    tokens = list(jieba.cut(word))
    print(f"{word} → {tokens}")
