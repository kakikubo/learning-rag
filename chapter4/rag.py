# RAG (Retrieval Augmented Generation)
system_prompt = '''\
文脈を踏まえて質問に1文で回答してください。

文脈: """
<LangChainのREADMEの内容>
"""

質問：{content}
'''

from openai import OpenAI

client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "LangGraphとは？"},
    ]
)
print(response.choices[0].message.content)
