# 上記と同様の結果をrole=systemを使って実現する 
from openai import OpenAI

client = OpenAI()

def generate_recipe(dish: str) -> str:
  response = client.chat.completions.create(
      model="gpt-3.5-turbo",
      messages=[
          {"role": "system", "content": "ユーザーが入力した料理のレシピを考えてください。"},
          {"role": "user", "content": f"{dish}"},
      ],
  )
  return response.choices[0].message.content

recipe = generate_recipe("カレー")
print(recipe)
