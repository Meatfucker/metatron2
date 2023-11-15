from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.utils import logging
import torch
from loguru import logger
import torchvision
from PIL import Image
import asyncio
import math
import io
from compel import Compel
import re

logging.set_verbosity_error()
logger.remove()




def format_prompt_weights(input_string):
    matches = re.findall(r'\((.*?)\:(.*?)\)', input_string)
    for match in matches:
        words = match[0].split()
        number = match[1]
        replacement = ' '.join(f"({word}){number}" for word in words)
        input_string = input_string.replace(f"({match[0]}:{match[1]})", replacement)
    return input_string
    
@logger.catch
def get_inputs(batch_size, prompt, compel_proc):
    generator = [torch.Generator("cuda").manual_seed(i) for i in range(batch_size)]
    prompt = format_prompt_weights(prompt)
    prompts = batch_size * [prompt]
    prompt_embeds = compel_proc(prompts)
    return {"prompt_embeds": prompt_embeds, "generator": generator}
    

@logger.catch
async def load_sd():
    model_id = "runwayml/stable-diffusion-v1-5"
    pipeline = DiffusionPipeline.from_pretrained(model_id, safety_checker=None, torch_dtype=torch.float16, use_safetensors=True)
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
    pipeline = pipeline.to("cuda")
    logger.success("SD Model Loaded.")
    compel_proc = Compel(tokenizer=pipeline.tokenizer, text_encoder=pipeline.text_encoder)
    return pipeline, compel_proc

@logger.catch
async def imagegenerate(pipeline, compel_proc, prompt, batch_size):
    logger.debug("IMAGEGEN Generate Started.")
    images = await asyncio.to_thread(pipeline, **get_inputs(batch_size, prompt, compel_proc), num_inference_steps=20)
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