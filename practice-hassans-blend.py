from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

model_id = "hassanblend/hassanblend1.4"

pipe = StableDiffusionPipeline.from_pretrained(model_id)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to("cuda")

prompt = "a cat"

image = pipe(
  prompt,
  num_inference_steps=25,
  guidance_scale=7.5,
  width=512,
  height=512
).images[0]

image.save("test.png")
