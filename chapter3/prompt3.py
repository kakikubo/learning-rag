# プロンプトのテンプレート化
prompt = '''\
以下の料理のレシピを考えてください。

料理名: """
{dish}
"""
'''

from openai import OpenAI

client = OpenAI()

def generate_recipe(dish: str) -> str:
  response = client.chat.completions.create(
      model="gpt-4o-mini",
      messages=[
          {"role": "user", "content": prompt.format(dish=dish)},
      ],
  )
  return response.choices[0].message.content

recipe = generate_recipe("カレー")
print(recipe)
