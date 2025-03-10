# Ragasによる合成テストデータの生成

# 検索対象のドキュメントのロード
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

# Ragasによる合成テストデータ生成の実装。Ragasが使用するメタデータである「filename」を設定
for document in documents:
  document.metadata["filename"] = document.metadata["source"]

# Ragasの機能で合成テストデータを生成(※数ドル程度の料金が発生します)
import nest_asyncio
from ragas.testset.generator import TestsetGenerator
from ragas.testset.evolutions import simple, reasoning, multi_context
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import time
from openai import RateLimitError

nest_asyncio.apply()

generator = TestsetGenerator.from_langchain(
    generator_llm=ChatOpenAI(model="gpt-4o-mini"),
    critic_llm=ChatOpenAI(model="gpt-4o-mini"),
    embeddings=OpenAIEmbeddings(),
)

def generate_testset_with_retries(generator, documents, retries=5):
    for i in range(retries):
        try:
            return generator.generate_with_langchain_docs(
                documents,
                test_size=4,
                distributions={simple: 0.5, reasoning: 0.25, multi_context: 0.25},
            )
        except RateLimitError as e:
            if i < retries - 1:
                wait_time = 2 ** i  # Exponential backoff
                print(f"Rate limit exceeded. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                raise e

testset = generate_testset_with_retries(generator, documents)

testset.to_pandas()

# LangSmithのDatasetの作成
from langsmith import Client

dataset_name = "agent-book"

client = Client()

if client.has_dataset(dataset_name=dataset_name):
  client.delete_dataset(dataset_name=dataset_name)

dataset = client.create_dataset(dataset_name=dataset_name)

# 合成テストデータの保存。LangSmithのDatasetに保存する形式に変換
inputs = []
outputs = []
metadatas = []

for testset_record in testset.test_data:
  inputs.append(
      {
          "question": testset_record.question,
      }
  )
  outputs.append(
      {
          "contexts": testset_record.contexts,
          "ground_truth": testset_record.ground_truth,
      }
  )
  metadatas.append(
      {
          "source": testset_record.metadata[0]["source"],
          "evolution_type": testset_record.evolution_type,
      }
  )

# LangSmithのクライアントを使用して、DatasetのIDを指定してデータを保存
client.create_examples(
    inputs=inputs,
    outputs=outputs,
    metadatas=metadatas,
    dataset_id=dataset.id,
)
