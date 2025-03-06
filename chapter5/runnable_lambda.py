# Runnable Lambdaのサンプル
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

from langchain_core.runnables import RunnableLambda

def upper(text: str) -> str:
  return text.upper()

chain = prompt | model | output_parser | RunnableLambda(upper)

output = chain.invoke({"input": "Hello!"})
print(output)
