from diffusers import AutoPipelineForInpainting, DEISMultistepScheduler
import torch
from diffusers.utils import load_image

pipe = AutoPipelineForInpainting.from_pretrained(
    "lykon-models/dreamshaper-8-inpainting", torch_dtype=torch.float16, variant="fp16"
)
pipe.scheduler = DEISMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to("cuda")

img_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo.png"
mask_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo_mask.png"

image = load_image(img_url)
mask_image = load_image(mask_url)


prompt = "a majestic tiger sitting on a park bench"

generator = torch.manual_seed(33)
image = pipe(
    prompt,
    image=image,
    mask_image=mask_image,
    generator=generator,
    num_inference_steps=25,
).images[0]
image.save("./image.png")
