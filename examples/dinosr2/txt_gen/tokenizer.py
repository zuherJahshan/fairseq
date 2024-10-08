from transformers import GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Example text to encode
text = "Hello, how are you doing today?"

# Encode the text
encoded_input = tokenizer.encode(text, return_tensors="pt")
print("Encoded input:", encoded_input)

# Decode the tokens back to text
decoded_output = tokenizer.decode(encoded_input[0], skip_special_tokens=True)
print("Decoded output:", decoded_output)

print("the vocabulary size of the tokenizer is:", tokenizer.vocab_size)
