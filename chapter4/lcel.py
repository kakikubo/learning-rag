# promptとmodelの連鎖
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "ユーザあが入力した料理のレシピを考えてください。"),
        ("human", "{dish}")
    ]
)

model = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

# promptとchainをつなぐ(LCEL)
chain = prompt | model

ai_message = chain.invoke({"dish": "カレー"})
print(ai_message.content)
