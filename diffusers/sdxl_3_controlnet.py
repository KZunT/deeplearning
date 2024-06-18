import torch
from diffusers import StableDiffusion3ControlNetPipeline
from diffusers.models import SD3ControlNetModel, SD3MultiControlNetModel
from diffusers.utils import load_image

# load pipeline
controlnet = SD3ControlNetModel.from_pretrained("InstantX/SD3-Controlnet-Canny")
pipe = StableDiffusion3ControlNetPipeline.from_pretrained(
    "stabilityai/stable-diffusion-3-medium-diffusers", controlnet=controlnet
)
pipe.to("cuda", torch.float16)

# config
control_image = load_image(
    "https://huggingface.co/InstantX/SD3-Controlnet-Canny/resolve/main/canny.jpg"
)
prompt = 'Anime style illustration of a girl wearing a suit. A moon in sky. In the background we see a big rain approaching. text "InstantX" on image'
n_prompt = "NSFW, nude, naked, porn, ugly"
image = pipe(
    prompt,
    negative_prompt=n_prompt,
    control_image=canny_image,
    controlnet_conditioning_scale=0.5,
).images[0]
image.save("image.jpg")
