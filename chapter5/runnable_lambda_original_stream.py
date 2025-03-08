# 独自の関数をstreamに対応させる
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from typing import Iterator

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        ("human", "{input}")
    ]
)

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

output_parser = StrOutputParser()

def upper(input_stream: Iterator[str]) -> Iterator[str]:
  for text in input_stream:
    yield text.upper()
  
chain = prompt | model | output_parser | upper

for chunk in chain.stream({"input": "Hello!"}):
  print(chunk, end="", flush=True)
