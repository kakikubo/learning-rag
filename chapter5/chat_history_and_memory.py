# Chat historyとMemory
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "{chat_history}"),
        ("human", "{input}"),
    ]
)

model = ChatOpenAI(model="gpt-4", temperature=0)

output_parser = StrOutputParser()
# Runnable Sequenceをつくる
chain = prompt | model | output_parser

def respond(session_id: str, human_message: str) -> str:
  chat_message_history = SQLChatMessageHistory(
      session_id=session_id, connection="sqlite:///sqlite.db"
  )
  messages = chat_message_history.get_messages()

  ai_message = chain.invoke(
      {
          "chat_history": messages,
          "input": human_message,
      }
  )

  chat_message_history.add_user_message(human_message)
  chat_message_history.add_ai_message(ai_message)

  return ai_message

# 呼び出し処理
from uuid import uuid4

session_id = uuid4().hex

output1 = respond(
    session_id=session_id,
    human_message="こんにちは！私はジョンと言います！",
)
print(output1)

output2 = respond(
    session_id=session_id,
    human_message="私の名前がわかりますか？"
)
print(output2)
