# Few-shotプロンプティング
from openai import OpenAI

client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "入力がAIに関係するか回答してください。"},
        {"role": "user", "content": "AIの進化はすごい"}, # Few-shotの例として回答例を記載する
        {"role": "assistant", "content": "true"},        # trueの例
        {"role": "user", "content": "今日は良い天気だ"},
        {"role": "assistant", "content": "false"},       # falseの例

        {"role": "user", "content": "ChatGPTはとても便利だ"},
    ]
)
print(response.choices[0].message.content)
