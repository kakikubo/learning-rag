# with_structured_outputを使った例(PydanticOutputParserよりも推奨)
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

class Recipe(BaseModel):
  ingredients: list[str] = Field(description="ingredients of the dish")
  steps: list[str] = Field(description="steps to make the dish")

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "ユーザーが入力した料理のレシピを考えてください。"),
        ("human", "{dish}"),
    ]
)

model = ChatOpenAI(model="gpt-4o-mini")

chain = prompt | model.with_structured_output(Recipe)

recipe = chain.invoke({"dish": "カレー"})
print(type(recipe))
print(recipe)
