from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.utils import make_image_grid
import torch
from loguru import logger
import torchvision
from PIL import Image
import asyncio
import math
import io

logger.remove()

@logger.catch
def get_inputs(batch_size, prompt):
    generator = [torch.Generator("cuda").manual_seed(i) for i in range(batch_size)]
    prompts = batch_size * [prompt]
    
    return {"prompt": prompts, "generator": generator}

@logger.catch
async def load_sd():
    model_id = "runwayml/stable-diffusion-v1-5"
    pipeline = DiffusionPipeline.from_pretrained(model_id, safety_checker=None, torch_dtype=torch.float16, use_safetensors=True)
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
    pipeline = pipeline.to("cuda")
    logger.success("SD Model Loaded.")
    return pipeline

@logger.catch
async def imagegenerate(pipeline, prompt, batch_size):
    logger.debug("IMAGEGEN Generate Started.")
    images = await asyncio.to_thread(pipeline, **get_inputs(batch_size, prompt), num_inference_steps=20)
    logger.debug("IMAGEGEN Generate Finished.")
    width, height = images.images[0].size
    num_images_per_row = math.ceil(math.sqrt(len(images.images)))
    num_rows = math.ceil(len(images.images) / num_images_per_row)
    composite_width = num_images_per_row * width
    composite_height = num_rows * height
    composite_image = Image.new('RGB', (composite_width, composite_height))
    for idx, image in enumerate(images.images):
        row, col = divmod(idx, num_images_per_row)
        composite_image.paste(image, (col * width, row * height))
    composite_image_bytes = io.BytesIO()
    composite_image.save(composite_image_bytes, format='PNG')
    composite_image_bytes.seek(0)
    return composite_image_bytes