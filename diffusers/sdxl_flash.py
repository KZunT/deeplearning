import torch
from diffusers import StableDiffusionXLPipeline, DPMSolverSinglestepScheduler

# Load model.
pipe = StableDiffusionXLPipeline.from_pretrained(
    "sd-community/sdxl-flash", torch_dtype=torch.float16, variant="fp16"
).to("cuda")

# Ensure sampler uses "trailing" timesteps.
pipe.scheduler = DPMSolverSinglestepScheduler.from_config(
    pipe.scheduler.config, timestep_spacing="trailing"
)

# Image generation.
pipe("a happy dog, sunny day, realism", num_inference_steps=7, guidance_scale=3).images[
    0
].save("output.png")
