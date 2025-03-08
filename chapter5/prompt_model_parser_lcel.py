# 今度はLCELをつかう。ChatPromptTemplate, ChatOpenAI, StrOutputParserはそれぞれRunnableを継承している
# Runnableを|でつなぐとRunnable Sequenceになる。
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
# Runnable Sequenceをinvokeすると連結したRunnableが順にinvokeされる
output = chain.invoke({"dish": "カレー"})
print(output)
