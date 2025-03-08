# LCELはprompt template, chat model, output parserを連結するのが基本
# これら3つを最初に用意し、それぞれでinvokeしてみる
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
# prompt, model, output parserを順にinvokeするコード
prompt_value = prompt.invoke({"dish": "カレー"})
ai_message = model.invoke(prompt_value)
output = output_parser.invoke(ai_message)
print(output)
