# Few-shotプロンプティングの別解3(Chat Completions API) 会話履歴でない事を強調
from openai import OpenAI

client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "入力がAIに関係するか回答してください。"},
        {"role": "system", "name": "example_user", "content": "AIの進化はすごい"},
        {"role": "system", "name": "example_assistant", "content": "true"},
        {"role": "system", "name": "example_user", "content": "今日は良い天気だ"},
        {"role": "system", "name": "example_assistant", "content": "false"},
        {"role": "user", "content": "ChatGPTはとても便利だ"},
    ]
)
print(response.choices[0].message.content)
