import json
import requests

headers = {"Content-Type": "application/json"}


def text_generation(data, ip="localhost", port=None):
    resp = requests.put(
        f"http://{ip}:{port}/generate", data=json.dumps(data), headers=headers
    )
    return resp.json()


def get_generation(
    prompt,
    greedy,
    add_BOS,
    token_to_gen,
    min_tokens,
    temp,
    top_p,
    top_k,
    repetition,
    batch=False,
):
    data = {
        "sentences": [prompt] if not batch else prompt,
        "tokens_to_generate": int(token_to_gen),
        "temperature": temp,
        "add_BOS": add_BOS,
        "top_k": top_k,
        "top_p": top_p,
        "greedy": greedy,
        "all_probs": False,
        "repetition_penalty": repetition,
        "min_tokens_to_generate": int(min_tokens),
        "end_strings": ["<|endoftext|>", "<extra_id_1>", "\x11", "<extra_id_1>User"],
    }
    sentences = text_generation(data, port=1424)["sentences"]
    return sentences[0] if not batch else sentences


PROMPT_TEMPLATE = """<extra_id_0>System

<extra_id_1>User
{prompt}
<extra_id_1>Assistant
"""

question = "Write a poem on NVIDIA in the style of Shakespeare"
prompt = PROMPT_TEMPLATE.format(prompt=question)
print(prompt)

response = get_generation(
    prompt,
    greedy=True,
    add_BOS=False,
    token_to_gen=1024,
    min_tokens=1,
    temp=1.0,
    top_p=1.0,
    top_k=0,
    repetition=1.0,
    batch=False,
)
response = response[len(prompt) :]
if response.endswith("<extra_id_1>"):
    response = response[: -len("<extra_id_1>")]
print(response)
