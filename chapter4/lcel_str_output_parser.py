# promptとmodelの連鎖(lcel.py)に加えてStrOutputParserを追加
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "ユーザあが入力した料理のレシピを考えてください。"),
        ("human", "{dish}")
    ]
)

model = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

output_parser = StrOutputParser()

# promptとmodelとoutput_parserをつなぐ(LCEL)
chain = prompt | model | output_parser
recipe = chain.invoke({"dish": "カレー"})
print(recipe)
