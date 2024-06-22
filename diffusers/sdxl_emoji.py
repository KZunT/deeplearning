import torch
from huggingface_hub import hf_hub_download
from diffusers import DiffusionPipeline
from cog_sdxl.dataset_and_utils import TokenEmbeddingsHandler
from diffusers.models import AutoencoderKL

pipe = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    variant="fp16",
).to("cuda")

pipe.load_lora_weights("fofr/sdxl-emoji", weight_name="lora.safetensors")

text_encoders = [pipe.text_encoder, pipe.text_encoder_2]
tokenizers = [pipe.tokenizer, pipe.tokenizer_2]

embedding_path = hf_hub_download(
    repo_id="fofr/sdxl-emoji", filename="embeddings.pti", repo_type="model"
)
embhandler = TokenEmbeddingsHandler(text_encoders, tokenizers)
embhandler.load_embeddings(embedding_path)
prompt = "A <s0><s1> emoji of a man"
images = pipe(
    prompt,
    cross_attention_kwargs={"scale": 0.8},
).images
# your output image
images[0]
