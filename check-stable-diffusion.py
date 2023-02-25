import torch
from diffusers import StableDiffusionPipeline
from torch import autocast

prompt = "a cat"
YOUR_TOKEN = "【Hugging Face API Key】"
MODEL_ID = "CompVis/stable-diffusion-v1-4"
DEVICE = "cuda"

pipe = StableDiffusionPipeline.from_pretrained(MODEL_ID, revision="fp16", torch_dtype=torch.float16, use_auth_token=YOUR_TOKEN)
pipe.to(DEVICE)
# Avoid Safety Checker : Start
def null_safety(images, **kwargs):
  return images, False
pipe.safety_checker = null_safety
# Avoid Safety Checker : End
with autocast(DEVICE):
  image = pipe(prompt, guidance_scale=7.5)["images"][0]
  image.save("test.png")
