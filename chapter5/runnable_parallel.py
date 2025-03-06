# RunnableParallel - 複数のRunnableを並列につなげる
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
output_parser = StrOutputParser()

# 楽観的な意見を生成するChain
optimistic_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "あなたは楽観主義者です。ユーザの入力に対して楽観的な意見をください。"),
        ("human", "{topic}"),
    ]
)
optimistic_chain = optimistic_prompt | model | output_parser

# 悲観的な意見を生成するChain
pessimistic_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "あなたは悲観主義者です。ユーザーの入力に対して悲観的な意見をください。"),
        ("human", "{topic}")
    ]
)
pessimistic_chain = pessimistic_prompt | model | output_parser

import pprint
from langchain_core.runnables import RunnableParallel

parallel_chain = RunnableParallel(
    {
        "optimistic_opinion": optimistic_chain,
        "pessimistic_opinion": pessimistic_chain
    }
)

output = parallel_chain.invoke({"topic": "生成AIの進化について"})
pprint.pprint(output)
