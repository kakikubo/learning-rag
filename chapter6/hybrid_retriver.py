# ハイブリッド検索 - 埋め込みベクトルの類似度検索Retriever(chroma_retriever)とBM25を使った検索のRetriever(bm25_retriever)で検索
# {{{ 基本部分
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

# RRF
from langchain_core.documents import Document

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



# }}}
from langchain_community.retrievers import BM25Retriever

chroma_retriever = retriever.with_config(
    {"run_name": "chroma_retriever"}
)

bm25_retriever = BM25Retriever.from_documents(documents).with_config(
    {"run_name": "bm25_retriever"}
)

# 両方の検索をするhybrid_retrieverを実装
from langchain_core.runnables import RunnableParallel

hybrid_retriever = (
    RunnableParallel({
        "chroma_documents": chroma_retriever,
        "bm25_documents": bm25_retriever,
    })
    | (lambda x: [x["chroma_documents"], x["bm25_documents"]])
    | reciprocal_rank_fusion
)

hybrid_rag_chain = (
    {
        "question": RunnablePassthrough(),
        "context": hybrid_retriever,
    }
    | prompt | model | StrOutputParser()
)

hybrid_rag_chain.invoke("LangChainの概要を教えて")
