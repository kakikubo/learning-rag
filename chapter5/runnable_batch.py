# Runnableをバッチ実行で複数の入力を扱う(batch)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "ユーザーが入力した料理のレシピを考えてください。"),
        ("human", "{dish}"),
    ]
)

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

output_parser = StrOutputParser()
# Runnable Sequenceをつくる
chain = prompt | model | output_parser

outputs = chain.batch([{"dish": "カレー"}, {"dish": "うどん"}])
print(outputs)
