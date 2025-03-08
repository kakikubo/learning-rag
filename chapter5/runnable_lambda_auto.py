# RunnableLambdaへの自動変換
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        ("human", "{input}")
    ]
)

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

output_parser = StrOutputParser()

def upper(text: str) -> str:
  return text.upper()

# upper関数にRunnableLambdaや@chainを使わなくとも、`|`の左右どちらかがRunnableの場合、もう一方が関数であれば自動的にRunnableLambdaに変換される
# chain = prompt | model | upper # 出力と入力があわずエラーになる。AttributeError: 'AIMessage' object has no attribute 'upper'
# chain = prompt | model | output_parser | upper # ok
chain = prompt | model | StrOutputParser() | upper # ok 。上と同じ

output = chain.invoke({"input": "Hello!"})
print(output)
