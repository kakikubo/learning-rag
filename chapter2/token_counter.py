import tiktoken

text = "LLMを使ってクールなものを作るのは簡単だが、プロダクションで使えるものを作るのは非常に難しい。"

encoding = tiktoken.encoding_for_model("gpt-4o")
tokens = encoding.encode(text)
print(len(tokens))
