from transformers import pipeline, set_seed

generator = pipeline("text-generation", model="distilgpt2")
set_seed(42)
generator("Hello, I’m a language model", max_length=20, num_return_sequences=5)


from transformers import GPT2Tokenizer, GPT2Model

tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
model = GPT2Model.from_pretrained("distilgpt2")
text = "Replace me by any text you'd like."
encoded_input = tokenizer(text, return_tensors="pt")
output = model(**encoded_input)
