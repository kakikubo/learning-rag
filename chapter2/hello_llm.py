from openai import OpenAI

client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "こんにちは!わたしはジョンと言います!"},
        {"role": "assistant", "content": "こんにちは、ジョンさん! お会いできて嬉しいです。今日はどんなことをお話ししましょうか？"},
        {"role": "user", "content": "私の名前がわかりますか？"},
    ],
)
print(response.to_json(indent=2))
