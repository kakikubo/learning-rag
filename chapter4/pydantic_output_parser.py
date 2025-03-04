# PydanticOutputParserを使ったPythonオブジェクトへの変換
from pydantic import BaseModel, Field

class Recipe(BaseModel):
  ingredients: list[str] = Field(description="ingredients of the dish")
  steps: list[str] = Field(description="steps to make the dish")

from langchain_core.output_parsers import PydanticOutputParser

output_parser = PydanticOutputParser(pydantic_object=Recipe)

format_instructions = output_parser.get_format_instructions()
# print(format_instructions)

# format_instructionsをつかったChatPromptTemplateを作成
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "ユーザーが入力した料理のレシピを考えてください。\n\n"
            "{format_instructions}",
        ),
        ("human", "{dish}"),
    ]
)

prompt_with_format_instructions = prompt.partial(
    format_instructions=format_instructions
)

# ChatPromptTemplateに対して例として入力を与えてみる
prompt_value = prompt_with_format_instructions.invoke({"dish": "カレー"})
print("=== role: system ===")
print(prompt_value.messages[0].content)
print("=== role: user ===")
print(prompt_value.messages[1].content)

# Recipeクラスの定義をもとに出力形式を指定するプロンプトが自動で埋め込まれている
# このテキストを入力としてLLMを実行する
from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

ai_message = model.invoke(prompt_value)
print(ai_message.content)

# この応答をPydanticのモデルのインスタンスに変換して使いたいため、PydanticOutputParserを使う
recipe = output_parser.invoke(ai_message)
print(type(recipe))
print(recipe)
