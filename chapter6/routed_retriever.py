# 質問に応じてWeb検索(Tavily)とLangChain公式ドキュメントを使い分ける

# ドキュメントローダー
from langchain_community.document_loaders import GitLoader

def file_filter(file_path: str) -> bool:
  return file_path.endswith(".mdx")

loader = GitLoader(
    clone_url="https://github.com/langchain-ai/langchain",
    repo_path="./langchain",
    branch="master",
    file_filter=file_filter,
)

documents = loader.load()

# ベクトル化
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
db = Chroma.from_documents(documents, embeddings)

# Retriever
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

prompt = ChatPromptTemplate.from_template('''\
以下の文脈だけを踏まえて質問に回答してください。

文脈: """
{context}
"""

質問: {question}
''')

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
retriever = db.as_retriever()

# ドキュメント検索とWeb検索を使い分ける
from langchain_community.retrievers import TavilySearchAPIRetriever

langchain_document_retriever = retriever.with_config(
    {"run_name": "langchain_document_retriever"}
)

web_retriever = TavilySearchAPIRetriever(k=3).with_config(
    {"run_name": "web_retriever"}
)
# route_chainの実装
from enum import Enum
from pydantic import BaseModel

class Route(str, Enum):
  langchain_document = "langchain_document"
  web = "web"

class RouteOutput(BaseModel):
  route: Route

route_prompt = ChatPromptTemplate.from_template("""\
質問に回答するために適切なRetrieverを選択してください。

質問: {question}
""")

route_chain = (
    route_prompt
    | model.with_structured_output(RouteOutput)
    | (lambda x: x.route)
)

# ルーティングの結果を踏まえて検索するrouted_retriever
from typing import Any
from langchain_core.documents import Document

def routed_retriever(inp: dict[str, Any]) -> list[Document]:
  question = inp["question"]
  route = inp["route"]

  if route == Route.langchain_document:
    return langchain_document_retriever.invoke(question)
  elif route == Route.web:
    return web_retriever.invoke(question)
  
  raise ValueError(f"Unknown route: {route}")

route_rag_chain = (
    {
        "question": RunnablePassthrough(),
        "route": route_chain,
    }
    | RunnablePassthrough.assign(context=routed_retriever)
    | prompt | model | StrOutputParser()
)

route_rag_chain.invoke("LangChainの概要を教えて")
