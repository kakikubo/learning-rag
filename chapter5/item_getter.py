# RunnableLambdaとの組み合わせ -- itemgetterを使う例
from operator import itemgetter
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

# 楽観的意見と悲観的意見をまとめるChain
synthesize_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "あなたは客観的AIです。{topic}について2つの意見をまとめてください。",
        ),
        (
            "human",
            "楽観的意見: {optimistic_opinion}\n悲観的意見: {pessimistic_opinion}"
        ),
    ]
)

synthesize_chain = (
    {
        "optimistic_opinion": optimistic_chain,
        "pessimistic_opinion": pessimistic_chain,
        "topic": itemgetter("topic")
    }
    | synthesize_prompt
    | model
    | output_parser
)

output = synthesize_chain.invoke({"topic": "生成AIの進化について"})
print(output)
