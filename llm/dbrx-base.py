from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

tokenizer = AutoTokenizer.from_pretrained(
    "databricks/dbrx-base", trust_remote_code=True, token="hf_YOUR_TOKEN"
)
model = AutoModelForCausalLM.from_pretrained(
    "databricks/dbrx-base",
    device_map="cpu",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    token="hf_YOUR_TOKEN",
)

input_text = "Databricks was founded in "
input_ids = tokenizer(input_text, return_tensors="pt")

outputs = model.generate(**input_ids, max_new_tokens=100)
print(tokenizer.decode(outputs[0]))
