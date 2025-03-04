# LangChainからChat modelを利用してOpenAIのChat Completion APIを呼び出す
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

messages = [
    SystemMessage("You are a helpful assistant."),
    HumanMessage("こんにちは!私はジョンと言います!"),
    AIMessage(content="こんにちは、ジョンさん!どのようにお手伝いできますか？"),
    HumanMessage(content="私の名前がわかりますか？"),
]

ai_message = model.invoke(messages) # 内部的には第2章の「Few-shotプロンプティング」で出てきたあたりのリクエストが送信されている
print(ai_message.content)
