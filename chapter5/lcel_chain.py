# LCELの「|」でさまざまなRunnableを連鎖させる
# Zero-shot CoTでステップバイステップで考えさせて、その結果から結論だけを抽出させてみる
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

output_parser = StrOutputParser()
# 1つ目のchainではZero-shot CoTでステップバイステップで考えさせる
cot_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "ユーザーの質問にステップバイステップで回答してください。"),
        ("human", "{question}")
    ]
)
cot_chain = cot_prompt | model | output_parser

# 2つ目のchainではステップバイステップで考えた回答から結論を抽出する
summarize_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "ステップバイステップで考えた回答から結論だけ抽出してください"),
        ("human", "{text}"),
    ]
)
summarize_chain = summarize_prompt | model | output_parser

# 2つのchainをつなげる
cot_summarize_chain = cot_chain | summarize_chain
cot_summarize_chain.invoke({"question": "10 + 2 * 3"})
