# RAGの基本形にHyDEを適用
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
print(len(documents))

# ベクトル化
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
db = Chroma.from_documents(documents, embeddings)


# RAGのChain
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

# HyDE (Hypothetical Document Embeddings)
hypothetical_prompt = ChatPromptTemplate.from_template("""\
次の質問に回答する一文を書いてください。

質問: {question}
""")

hypothetical_chain = hypothetical_prompt | model | StrOutputParser() # 仮説的な回答まで生成するChain

# 「仮説的な回答をするChain」を使ったRAGのChain
hyde_rag_chain = {
    "question": RunnablePassthrough(),
    "context": hypothetical_chain | retriever,
} | prompt | model | StrOutputParser()

hyde_output = hyde_rag_chain.invoke("LangChainの概要を教えて")
print(hyde_output)
