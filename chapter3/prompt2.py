# もう少し短い文章で回答してもらう
from openai import OpenAI

client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "質問に100文字程度で答えてください。"},
        {"role": "user", "content": "プロンプトエンジニアリングとは"},
    ],
)
print(response.choices[0].message.content)
