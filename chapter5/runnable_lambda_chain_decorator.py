# chainデコレーターをつかったRunnableLambdaの実装
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import chain

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        ("human", "{input}")
    ]
)

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

output_parser = StrOutputParser()

@chain
def upper(text: str) -> str:
  return text.upper()

chain = prompt | model | output_parser | upper

output = chain.invoke({"input": "Hello!"})
print(output)
