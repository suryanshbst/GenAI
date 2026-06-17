import tiktoken

encoder = tiktoken.encoding_for_model("gpt-4o")

text = "Hello, I am Suryansh"
tokens = encoder.encode(text)

print("Token: ", tokens)

tokens = [13225, 11, 357, 939, 336, 4248, 616, 71]
decoded = encoder.decode(tokens)

print("Decoded Text: ", decoded)