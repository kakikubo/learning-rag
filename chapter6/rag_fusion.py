# 検索後の工夫にRAG-Fusionを使い、複数の検索結果を並べ替えることができる。mutti_query.pyから改修する
from langchain_community.document_loaders import GitLoader
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

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

# 複数の検索クエリを生成するChain
class QueryGenerationOutput(BaseModel):
  queries: list[str] = Field(..., description="検索クエリのリスト")

query_generation_prompt = ChatPromptTemplate.from_template("""\
質問に対してベクターデータベースから関連文書を検索するために、
3つの異なる検索クエリを生成してください。
距離ベースの類似性検索の限界を克服するために、
ユーザーの質問に対して複数の指定を提供することが目標です。

質問: {question}
""")

query_generation_chain = (
    query_generation_prompt
    | model.with_structured_output(QueryGenerationOutput)
    | (lambda x: x.queries)
)

# RAG-Fusion
from langchain_core.documents import Document


# RRF
def reciprocal_rank_fusion(
        retriever_outputs: list[list[Document]],
        k: int = 60,
) -> list[str]:
    # 各ドキュメントのコンテンツ(文字列)とそのスコアの対応を保持する辞書を準備
    content_score_mapping = {}

    # 検索クエリごとにループ
    for docs in retriever_outputs:
        # 検索結果のドキュメントごとにループ
        for rank, doc in enumerate(docs):
            content = doc.page_content

            # 初めて登場したコンテンツの場合はスコアを0で初期化
            if content not in content_score_mapping:
                content_score_mapping[content] = 0

            # (1 / (順位 + k))のスコアを加算
            content_score_mapping[content] += 1 / (rank + k)

    # スコアの大きい順にソート
    ranked = sorted(content_score_mapping.items(), key=lambda x: x[1], reverse=True)
    return [content for content, _ in ranked]


# 上記関数をつかって、RAG-FusionのChain全体を実装
rag_fusion_chain = {
                       "question": RunnablePassthrough(),
                       "context": query_generation_chain | retriever.map() | reciprocal_rank_fusion,
                   } | prompt | model | StrOutputParser()

output = rag_fusion_chain.invoke("LangChainの概要を教えて")
print(output)
