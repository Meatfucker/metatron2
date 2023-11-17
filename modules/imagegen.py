#speakgen.py - Functions for stable diffusion capabilities
import asyncio
import io
import math
import os
import re
import random
import gc
import torch
import torchvision
from loguru import logger
from PIL import Image
from diffusers import DiffusionPipeline, StableDiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.utils import logging as difflogging
from transformers.utils import logging as translogging
from compel import Compel
import discord
from discord import app_commands
from modules.settings import SETTINGS

difflogging.set_verbosity_error() #Attempt to silence noisy diffusers log messages
translogging.set_verbosity_error() #Attempt to silence noisy transformers log messages
logger.remove() #attempt to silence noisy library log messages

@logger.catch
def format_prompt_weights(input_string):
    '''This takes a prompt, checks for A1111 style prompt weightings, and converts them to compel style weightings'''
    matches = re.findall(r'\((.*?)\:(.*?)\)', input_string)
    for match in matches:
        words = match[0].split()
        number = match[1]
        replacement = ' '.join(f"({word}){number}" for word in words)
        input_string = input_string.replace(f"({match[0]}:{match[1]})", replacement)
    return input_string

@logger.catch
def moderate_prompt(prompt, negativeprompt):
    '''Removes all the words in the imagebannedwords setting from prompt, and adds all the words in imagenegprompt to negativeprompt'''
    banned_words = SETTINGS["imagebannedwords"][0].split(',')
    for word in banned_words:
        prompt = prompt.replace(word.strip(), '')
    imagenegprompt_string = SETTINGS["imagenegprompt"][0]
    if negativeprompt == None:
        negativeprompt = ""
    negativeprompt = imagenegprompt_string + " " + negativeprompt
    return prompt, negativeprompt

@logger.catch
def get_inputs(batch_size, prompt, negativeprompt, compel_proc, seed):
    '''This multiples the prompt by the batch size and creates the weight embeddings'''
    if seed == None:
        random_seed = random.randint(-2147483648, 2147483647)
        generator = [torch.Generator("cuda").manual_seed(random.randint(-2147483648, 2147483647)) for i in range(batch_size)] #push the generators to gpu
    else:
        generator = [torch.Generator("cuda").manual_seed(seed + i) for i in range(batch_size)] #push the generators to gpu
    prompt, negativeprompt = moderate_prompt(prompt, negativeprompt)
    prompt = format_prompt_weights(prompt)
    prompts = batch_size * [prompt]
    with torch.no_grad():
        prompt_embeds = compel_proc(prompts)
    if negativeprompt != None:
        negativeprompt = format_prompt_weights(negativeprompt)
        negativeprompts = batch_size * [negativeprompt]
        with torch.no_grad():
            negative_prompt_embeds = compel_proc(negativeprompts)
        return {"prompt_embeds": prompt_embeds, "negative_prompt_embeds": negative_prompt_embeds, "generator": generator}
    else:
        return {"prompt_embeds": prompt_embeds, "generator": generator}
    
@logger.catch  
async def load_models_list():
    '''Get list of models for user interface'''
    models = []
    models_list = os.listdir("models/")
    for models_file in models_list:
        if models_file.endswith(".safetensors"):
            models.append(models_file)
    return models

@logger.catch
async def load_loras_list():
    '''Get list of models for user interface'''
    loras = []
    loras_list = os.listdir("loras/")
    for loras_file in loras_list:
        if loras_file.endswith(".safetensors"):
            token_name = f"{loras_file[:-12]}"
            loras.append(token_name)
    return loras

@logger.catch 
async def load_embeddings_list():
    '''Get list of models for user interface'''
    embeddings = []
    embeddings_list = os.listdir("embeddings/")
    for embeddings_file in embeddings_list:
        if embeddings_file.endswith(".pt"):
            token_name = f"<{embeddings_file[:-3]}>"
            embeddings.append(token_name)
    return embeddings

@logger.catch
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
    logger.success("SD Model Loaded.")
    compel_proc = Compel(tokenizer=pipeline.tokenizer, text_encoder=pipeline.text_encoder) #create the compel processor object for the pipeline
    with torch.no_grad(): #clear gpu memory cache
        torch.cuda.empty_cache()
    gc.collect() #clear python memory
    return pipeline, compel_proc

@logger.catch   
async def load_sd_lora(pipeline, prompt):
    ''' This loads a lora and applies it to a pipeline'''
    sd_need_reload = False
    new_prompt = prompt
    loramatches = re.findall(r'<lora:([^:]+):([\d.]+)>', prompt)
    if loramatches:
        logger.debug("LORA Loading...")
        for match in loramatches:
            loraname, loraweight = match
            loraweight = float(loraweight)  # Convert y to a float if needed
            lorafilename = f'{loraname}.safetensors'
            pipeline.load_lora_weights("./loras", weight_name=lorafilename)
            pipeline.fuse_lora(lora_scale=loraweight)
        new_prompt = re.sub(r'<lora:([^\s:]+):([\d.]+)>', '', prompt)
        sd_need_reload = True
        with torch.no_grad(): #clear gpu memory cache
            torch.cuda.empty_cache()
        gc.collect() #clear python memory
    return pipeline, new_prompt, sd_need_reload

@logger.catch    
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

@logger.catch
async def sd_generate(pipeline, compel_proc, prompt, model, batch_size, negativeprompt, seed, steps, width, height):
    '''this generates the request, tiles the images, and returns them as a single image'''
    logger.debug("IMAGEGEN Generate Started.")
    generate_width = math.ceil(width / 8) * 8
    generate_height = math.ceil(height / 8) * 8
    images = await asyncio.to_thread(pipeline, **get_inputs(batch_size, prompt, negativeprompt, compel_proc, seed), num_inference_steps=steps, width=generate_width, height=generate_height) #do the generate in a thread so as not to lock up the bot client
    pipeline.unload_lora_weights()
    logger.debug("IMAGEGEN Generate Finished.")
    composite_image_bytes = await make_image_grid(images)
    return composite_image_bytes

@logger.catch
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
    
class Imagegenbuttons(discord.ui.View):
    '''Class for the ui buttons on speakgen'''

    def __init__(self, generation_queue, prompt, channel, sdmodel, batch_size, username, userid, negativeprompt, seed, steps, width, height, metatron_client):
        super().__init__()
        self.timeout = None #Disables the timeout on the buttons
        self.generation_queue = generation_queue
        self.userid = userid
        self.prompt = prompt
        self.channel = channel
        self.sdmodel = sdmodel
        self.batch_size = batch_size
        self.username = username
        self.userid = userid
        self.negativeprompt = negativeprompt
        self.seed = None
        self.steps = steps
        self.width = width
        self.height = height
        self.metatron_client = metatron_client
    
    @logger.catch
    @discord.ui.button(label='Reroll', emoji="🎲", style=discord.ButtonStyle.grey)
    async def reroll(self, interaction: discord.Interaction, button: discord.ui.Button):
        '''Rerolls last reply'''
        if self.userid == interaction.user.id:
            if await self.metatron_client.is_room_in_queue(self.userid) == True:
                await interaction.response.send_message("Rerolling...", ephemeral=True, delete_after=5)
                await self.generation_queue.put(('imagegenerate', interaction.user.id, self.prompt, interaction.channel, self.sdmodel, self.batch_size, self.username, self.negativeprompt, self.seed, self.steps, self.width, self.height))
            else:
                await interaction.response.send_message("Queue limit reached, please wait until your current gen or gens finish")
    
    @logger.catch
    @discord.ui.button(label='Mail', emoji="✉", style=discord.ButtonStyle.grey)
    async def dmimage(self, interaction: discord.Interaction, button: discord.ui.Button):
        '''DMs sound'''
        await interaction.response.send_message("DM'ing image...", ephemeral=True, delete_after=5)
        sound_bytes = await interaction.message.attachments[0].read()
        dm_channel = await interaction.user.create_dm()
        truncated_filename = self.prompt[:1000]
        await dm_channel.send(file=discord.File(io.BytesIO(sound_bytes), filename=f'{truncated_filename}.png'))
        speak_dm_logger = logger.bind(user=interaction.user.name, userid=interaction.user.id)
        speak_dm_logger.success("IMAGEGEN DM successful")

    @logger.catch
    @discord.ui.button(label='Delete', emoji="❌", style=discord.ButtonStyle.grey)
    async def delete_message(self, interaction: discord.Interaction, button: discord.ui.Button):
        '''Deletes message'''
        if self.userid == interaction.user.id:
            await interaction.message.delete()
        await interaction.response.send_message("Image deleted.", ephemeral=True, delete_after=5)
        speak_delete_logger = logger.bind(user=interaction.user.name, userid=interaction.user.id)
        speak_delete_logger.info("IMAGEGEN Delete")
    