# LangChainのLLMからOpenAIのCompletions APIを利用する
from langchain_openai import OpenAI

model = OpenAI(model="gpt-3.5-turbo-instruct", temperature=0) # temperatureが大きいほど出力がランダムに。小さいほど決定的になる
output = model.invoke("自己紹介してください。")
print(output)
