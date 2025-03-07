# astream_events - Chainの中間の値を出力する
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_community.retrievers import TavilySearchAPIRetriever
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables import RunnableParallel

prompt = ChatPromptTemplate.from_template('''\
以下の文脈だけを踏まえて質問に回答してください。

文脈: """
{context}
"""

質問: {question}
''')

model = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
retriever = TavilySearchAPIRetriever(k=3)
# ここからがrunnable_passthrough.pyとの差分
import pprint

# chain = {
#     "question": RunnablePassthrough(),
#     "context": retriever,
# } | RunnablePassthrough.assign(answer=prompt | model | StrOutputParser())
# 次のように書くことも可能
chain = (
    {
      "question": RunnablePassthrough(),
      "context": retriever,
    }
    | prompt
    | model
    | StrOutputParser()
)

import asyncio

async def main():
    async for event in chain.astream_events("東京の今日の天気は？", version="v2"):
      event_kind = event["event"]

      if event_kind == "on_retriever_end":
        print("=== 検索結果 ===")
        documents = event["data"]["output"]
        for document in documents:
           print(document)
      elif event_kind == "on_parser_start":
        print("=== 最終出力 ===")
      elif event_kind == "on_parser_stream":
        chunk = event["data"]["chunk"]
        print(chunk, end="", flush=True)

if __name__ == "__main__":
    asyncio.run(main())
