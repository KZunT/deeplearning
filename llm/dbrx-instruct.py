from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

tokenizer = AutoTokenizer.from_pretrained(
    "databricks/dbrx-instruct", trust_remote_code=True
)
model = AutoModelForCausalLM.from_pretrained(
    "databricks/dbrx-instruct",
    device_map="cpu",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)

input_text = "What does it take to build a great LLM?"
messages = [{"role": "user", "content": input_text}]
input_ids = tokenizer.apply_chat_template(
    messages,
    return_dict=True,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
)

outputs = model.generate(**input_ids, max_new_tokens=200)
print(tokenizer.decode(outputs[0]))
