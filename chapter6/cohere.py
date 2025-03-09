# Cohereのリランクモデルの導入
from langchain_community.document_loaders import GitLoader
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

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
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
db = Chroma.from_documents(documents, embeddings)


# RAGのChain
prompt = ChatPromptTemplate.from_template('''\
以下の文脈だけを踏まえて質問に回答してください。

文脈: """
{context}
"""

質問: {question}
''')

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
retriever = db.as_retriever()
# Cohereのリランクモデルの導入
from typing import Any

from langchain_cohere import CohereRerank
from langchain_core.documents import Document

def rerank(inp: dict[str, Any], top_n: int = 3) -> list[Document]:
  question = inp["question"]
  documents = inp["documents"]

  cohere_reranker = CohereRerank(model="rerank-multilingual-v3.0", top_n=top_n) # top_nはリランク結果の上位何件を返すか設定するパラメータ
  return cohere_reranker.compress_documents(documents=documents, query=question)

rerank_rag_chain = (
    {
        "question": RunnablePassthrough(),
        "documents": retriever,
    }
    | RunnablePassthrough.assign(context=rerank)
    | prompt | model | StrOutputParser()
)

rerank_rag_chain.invoke("LangChainの概要を教えて")
