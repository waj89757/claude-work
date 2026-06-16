# 1. 导入和配置
import dspy
from config import configure_dspy
lm = configure_dspy()

# 2. 定义一个最简单的任务：翻译
class Translate(dspy.Signature):
    """Translate English to Chinese."""
    english: str = dspy.InputField()
    chinese: str = dspy.OutputField()

# 3. 创建模块并调用
translator = dspy.Predict(Translate)

# 4. 试一下！
result = translator(english="Hello, how are you?")
print(result.chinese)

# 5. 再试一个
result = translator(english="The weather is beautiful today.")
print(result.chinese)

# 6. 换成 ChainOfThought，看推理过程
cot_translator = dspy.ChainOfThought(Translate)
result = cot_translator(english="Life is short, learn Python.")
print(f"推理: {result.reasoning}")
print(f"翻译: {result.chinese}")