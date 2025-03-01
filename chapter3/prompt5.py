# 出力形式を指定する
system_prompt = """\
ユーザーが入力した料理のレシピを考えてください。
出力は以下のJSON形式にしてください。

```
{
  "材料": ["材料1", "材料2"],
  "手順": ["手順1", "手順2"]
}
```
"""

from openai import OpenAI

client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "カレー"},
    ]
)
print(response.choices[0].message.content)
