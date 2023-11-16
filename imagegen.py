#speakgen.py - Functions for Bark capabilities
import asyncio
import io
import math
import os
import re
import torch
import torchvision
from loguru import logger
from PIL import Image
from diffusers import DiffusionPipeline, StableDiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.utils import logging as difflogging
from transformers.utils import logging as translogging
from compel import Compel

difflogging.set_verbosity_error() #Attempt to silence noisy diffusers log messages
translogging.set_verbosity_error() #Attempt to silence noisy transformers log messages
logger.remove() #attempt to silence noisy library log messages

def format_prompt_weights(input_string):
    '''This takes a prompt, checks for A1111 style prompt weightings, and converts them to compel style weightings'''
    matches = re.findall(r'\((.*?)\:(.*?)\)', input_string)
    for match in matches:
        words = match[0].split()
        number = match[1]
        replacement = ' '.join(f"({word}){number}" for word in words)
        input_string = input_string.replace(f"({match[0]}:{match[1]})", replacement)
    return input_string

def get_inputs(batch_size, prompt, compel_proc):
    '''This multiples the prompt by the batch size and creates the weight embeddings'''
    generator = [torch.Generator("cuda").manual_seed(i) for i in range(batch_size)] #push the generators to gpu
    prompt = format_prompt_weights(prompt)
    prompts = batch_size * [prompt]
    prompt_embeds = compel_proc(prompts)
    return {"prompt_embeds": prompt_embeds, "generator": generator}
  
async def load_models_list():
    '''Get list of models for user interface'''
    models = []
    models_list = os.listdir("models/")
    for models_file in models_list:
        if models_file.endswith(".safetensors"):
            models.append(models_file)
    return models

async def load_embeddings_list():
    '''Get list of models for user interface'''
    embeddings = []
    embeddings_list = os.listdir("embeddings/")
    for embeddings_file in embeddings_list:
        if embeddings_file.endswith(".pt"):
            token_name = f"<{embeddings_file[:-3]}>"
            embeddings.append(token_name)
    return embeddings

async def load_sd(model = None, pipeline = None):
    '''Load a sd model, returning the pipeline object and the compel processor object for the pipeline'''
    logger.debug("SD Model Loading...")
    if model != None:
        model_id = f'./models/{model}'
        pipeline = StableDiffusionPipeline.from_single_file(model_id, load_safety_checker=False, torch_dtype=torch.float16, use_safetensors=True) #This loads a checkpoint file
    else:
        model_id = "runwayml/stable-diffusion-v1-5"
        pipeline = DiffusionPipeline.from_pretrained(model_id, safety_checker=None, torch_dtype=torch.float16, use_safetensors=True) #This loads a huggingface based model, is the initial loading model for now.
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config) #This is the sampler, I may make it configurable in the future
    pipeline = pipeline.to("cuda") #push the pipeline to gpu
    logger.debug("SD Model Loaded.")
    compel_proc = Compel(tokenizer=pipeline.tokenizer, text_encoder=pipeline.text_encoder) #create the compel processor object for the pipeline
    return pipeline, compel_proc
  
async def load_ti(pipeline, prompt, loaded_image_embeddings):
    '''this checks the prompt for textual inversion embeddings and loads them if needed, keeping track of used ones so it doesnt try to load the same one twice'''
    matches = re.findall(r'<(.*?)>', prompt)
    for match in matches:
        file_path = f"./embeddings/{match}.pt"
        token_name = f"<{match}>"
        if match not in loaded_image_embeddings:
            if os.path.exists(file_path):
                pipeline.load_textual_inversion(file_path, token=token_name) #this applies the embedding to the pipeline
                loaded_image_embeddings.append(match) #add the embedding to the current loaded list
    return pipeline, loaded_image_embeddings


async def sd_generate(pipeline, compel_proc, prompt, model, batch_size):
    '''this generates the request, tiles the images, and returns them as a single image'''
    logger.debug("IMAGEGEN Generate Started.")
    images = await asyncio.to_thread(pipeline, **get_inputs(batch_size, prompt, compel_proc), num_inference_steps=20) #do the generate in a thread so as not to lock up the bot client
    logger.debug("IMAGEGEN Generate Finished.")
    composite_image_bytes = await make_image_grid(images)
    return composite_image_bytes

async def make_image_grid(images):
    '''This takes a list of pil image objects and turns them into a grid'''
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
    