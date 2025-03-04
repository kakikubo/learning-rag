# LamgSmithのPrompts
# <https://smith.langchain.com/hub/oshima/recipe?organizationId=389973b1-cc16-4ceb-af89-a37de26f99b2>
from langsmith import Client

client = Client()
prompt = client.pull_prompt("oshima/recipe")

prompt_value = prompt.invoke({"dish": "カレー"})
print(prompt_value)
