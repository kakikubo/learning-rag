# PromptTemplate
from langchain_core.prompts import PromptTemplate

prompt = PromptTemplate.from_template("""以下の料理のレシピを考えてください。

料理名: {dish}""")

prompt_value = prompt.invoke({"dish": "カレー"})
print(prompt_value.text)
