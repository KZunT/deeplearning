from mistral_inference.transformer import Transformer
from mistral_inference.generate import generate

from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.protocol.instruct.messages import UserMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest

tokenizer = MistralTokenizer.from_file(f"{mistral_models_path}/tokenizer.model.v3")
model = Transformer.from_folder(mistral_models_path)

prompt = "How expensive would it be to ask a window cleaner to clean all windows in Paris. Make a reasonable guess in US Dollar."

completion_request = ChatCompletionRequest(messages=[UserMessage(content=prompt)])

tokens = tokenizer.encode_chat_completion(completion_request).tokens

out_tokens, _ = generate(
    [tokens],
    model,
    max_tokens=64,
    temperature=0.7,
    eos_id=tokenizer.instruct_tokenizer.tokenizer.eos_id,
)
result = tokenizer.decode(out_tokens[0])

print(result)
