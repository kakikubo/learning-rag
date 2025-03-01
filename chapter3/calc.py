# 数式を回答する(誤答する例: 8が正解だが、10と答えてしまう)
from openai import OpenAI

client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "回答だけ一言で出力してください。"},
        {"role": "user", "content": "10 + 2 * 3 - 4 * 2"},
    ]
)
print(response.choices[0].message.content) # => 10 (誤り)
