# imagegen.py - Functions for stable diffusion capabilities
import asyncio
import io
import math
import os
import re
import random
import gc
import diffusers.utils.logging
import torch
from loguru import logger
from PIL import Image
from diffusers import DiffusionPipeline, StableDiffusionPipeline, DPMSolverSinglestepScheduler
from compel import Compel
import discord
from modules.settings import SETTINGS, get_defaults


diffusers.utils.logging.set_verbosity_error()


def format_prompt_weights(input_string):
    """This takes a prompt, checks for A1111 style prompt weightings, and converts them to compel style weightings"""
    matches = re.findall(r'\((.*?)\:(.*?)\)', input_string)
    for match in matches:
        words = match[0].split()
        number = match[1]
        replacement = ' '.join(f"({word}){number}" for word in words)
        input_string = input_string.replace(f"({match[0]}:{match[1]})", replacement)
    return input_string


async def moderate_prompt(prompt, negativeprompt):
    """Removes all words in the imagebannedwords from prompt, adds all words in imagenegprompt to negativeprompt"""
    sd_defaults = await get_defaults('global')  # load global defaults
    banned_words = sd_defaults["imagebannedwords"][0].split(',')
    for word in banned_words:
        prompt = prompt.replace(word.strip(), '')
    imagenegprompt_string = sd_defaults["imagenegprompt"][0]
    if negativeprompt is None:
        negativeprompt = ""
    negativeprompt = negativeprompt + " " + imagenegprompt_string
    return prompt, negativeprompt


async def get_inputs(batch_size, prompt, negativeprompt, compel_proc, seed):
    """This multiples the prompt by the batch size and creates the weight embeddings"""
    if seed is None:  # Use a random seed if one isnt supplied
        generator = [torch.Generator("cuda").manual_seed(random.randint(-2147483648, 2147483647)) for _ in range(batch_size)]
    else:
        generator = [torch.Generator("cuda").manual_seed(seed + i) for i in range(batch_size)]
    prompt, negativeprompt = await moderate_prompt(prompt, negativeprompt)  # Moderate prompt according to settings.
    logger.debug(f'PROMPT: {prompt}')
    logger.debug(f'NEG: {negativeprompt}')
    prompt = format_prompt_weights(prompt)  # Apply compel weights
    prompts = batch_size * [prompt]
    with torch.no_grad():
        prompt_embeds = compel_proc(prompts)  # Make compel embeddings
    if negativeprompt is not None:
        negativeprompt = format_prompt_weights(negativeprompt)
        negativeprompts = batch_size * [negativeprompt]
        with torch.no_grad():
            negative_prompt_embeds = compel_proc(negativeprompts)
        return {"prompt_embeds": prompt_embeds, "negative_prompt_embeds": negative_prompt_embeds, "generator": generator}
    else:
        return {"prompt_embeds": prompt_embeds, "generator": generator}


async def load_models_list():
    """Get list of models for user interface"""
    models = []
    models_dir = "models/"
    with os.scandir(models_dir) as models_list:
        for models_file in models_list:
            if models_file.name.endswith(".safetensors"):
                models.append(models_file.name)
    return models


async def load_loras_list():
    """Get list of Loras for user interface"""
    loras = []
    loras_dir = "loras/"
    with os.scandir(loras_dir) as loras_list:
        for loras_file in loras_list:
            if loras_file.name.endswith(".safetensors"):
                token_name = f"{loras_file.name[:-12]}"
                loras.append(token_name)
    return loras



async def load_embeddings_list():
    """Get list of embeddings for user interface"""
    embeddings = []
    embeddings_dir = "embeddings/"
    with os.scandir(embeddings_dir) as embeddings_list:
        for embeddings_file in embeddings_list:
            if embeddings_file.name.endswith(".pt"):
                token_name = f"<{embeddings_file.name[:-3]}>"
                embeddings.append(token_name)
    return embeddings


async def load_sd(model=None):
    """Load a sd model, returning the pipeline object and the compel processor object for the pipeline"""
    logger.debug("SD Model Loading...")
    if model is not None:  # This loads a checkpoint file
        model_id = f'./models/{model}'
        pipeline = await asyncio.to_thread(StableDiffusionPipeline.from_single_file, model_id, load_safety_checker=False, torch_dtype=torch.float16, use_safetensors=True)
    else:
        sd_model_list = await load_models_list()
        if sd_model_list is not None:   # This loads a checkpoint file
            model_id = f'./models/{sd_model_list[0]}'
            pipeline = await asyncio.to_thread(StableDiffusionPipeline.from_single_file, model_id, load_safety_checker=False, torch_dtype=torch.float16, use_safetensors=True)
        else:  # This loads a huggingface based model, is the initial loading model for now.
            model_id = "runwayml/stable-diffusion-v1-5"
            pipeline = await asyncio.to_thread(DiffusionPipeline.from_pretrained, model_id, safety_checker=None, torch_dtype=torch.float16, use_safetensors=True)
    pipeline.scheduler = DPMSolverSinglestepScheduler.from_config(pipeline.scheduler.config)  # This is the sampler, I may make it configurable in the future
    pipeline = pipeline.to("cuda")  # push the pipeline to gpu
    load_sd_logger = logger.bind(model=model_id)
    load_sd_logger.success("SD Model Loaded.")
    compel_proc = Compel(tokenizer=pipeline.tokenizer, text_encoder=pipeline.text_encoder, truncate_long_prompts=False)  # create the compel processor object for the pipeline
    with torch.no_grad():  # clear gpu memory cache
        torch.cuda.empty_cache()
    gc.collect()  # clear python memory
    return pipeline, compel_proc, model


async def load_sd_lora(pipeline, prompt):
    """ This loads a lora and applies it to a pipeline"""
    new_prompt = prompt
    loramatches = re.findall(r'<lora:([^:]+):([\d.]+)>', prompt)
    if loramatches:
        names = []
        weights = []
        logger.debug("LORA Loading...")
        for match in loramatches:
            loraname, loraweight = match
            loraweight = float(loraweight)  # Convert to a float if needed
            lorafilename = f'{loraname}.safetensors'
            pipeline.load_lora_weights("./loras", weight_name=lorafilename, adapter_name=loraname)
            names.append(loraname)
            weights.append(loraweight)
        pipeline.set_adapters(names, adapter_weights=weights)
        new_prompt = re.sub(r'<lora:([^\s:]+):([\d.]+)>', '', prompt)  # This removes the lora trigger from the users prompt so it doesnt effect the gen.
        with torch.no_grad():  # clear gpu memory cache
            torch.cuda.empty_cache()
        gc.collect()  # clear python memory
    return pipeline, new_prompt


async def load_ti(pipeline, prompt, loaded_image_embeddings):
    """this checks the prompt for textual inversion embeddings and loads them if needed, keeping track of used ones so it doesnt try to load the same one twice"""
    matches = re.findall(r'<(.*?)>', prompt)
    for match in matches:
        file_path = f"./embeddings/{match}.pt"
        token_name = f"<{match}>"
        if match not in loaded_image_embeddings:
            if os.path.exists(file_path):
                pipeline.load_textual_inversion(file_path, token=token_name)  # this applies the embedding to the pipeline
                loaded_image_embeddings.append(match)  # add the embedding to the current loaded list
    return pipeline, loaded_image_embeddings


async def sd_generate(pipeline, compel_proc, prompt, model, batch_size, negativeprompt, seed, steps, width, height):
    """this generates the request, tiles the images, and returns them as a single image"""
    if batch_size > int(SETTINGS["maxbatch"][0]):  # These lines ensure compliance with the maxbatch and maxres settings.
        batch_size = int(SETTINGS["maxbatch"][0])
    if width > int(SETTINGS["maxres"][0]):
        width = int(SETTINGS["maxres"][0])
    if height > int(SETTINGS["maxres"][0]):
        height = int(SETTINGS["maxres"][0])
    generate_width = math.ceil(width / 8) * 8  # Dimensions have to be multiple of 8 or else SD shits itself.
    generate_height = math.ceil(height / 8) * 8
    inputs = await get_inputs(batch_size, prompt, negativeprompt, compel_proc, seed)  # This creates the prompt embeds
    pipeline.set_progress_bar_config(disable=True)
    sd_generate_logger = logger.bind(prompt=prompt, negative_prompt=negativeprompt, model=model)
    sd_generate_logger.debug("IMAGEGEN Generate Started.")
    with torch.no_grad():  # do the generate in a thread so as not to lock up the bot client, and no_grad to save memory.
        images = await asyncio.to_thread(pipeline, **inputs, num_inference_steps=steps, width=generate_width, height=generate_height)  # do the generate in a thread so as not to lock up the bot client
    pipeline.unload_lora_weights()
    sd_generate_logger.debug("IMAGEGEN Generate Finished.")
    composite_image_bytes = await make_image_grid(images)  # Turn returned images into a single image
    return composite_image_bytes


async def make_image_grid(images):
    """This takes a list of pil image objects and turns them into a grid"""
    width, height = images.images[0].size  # Figure out final image dimensions.
    num_images_per_row = math.ceil(math.sqrt(len(images.images)))
    num_rows = math.ceil(len(images.images) / num_images_per_row)
    composite_width = num_images_per_row * width
    composite_height = num_rows * height
    composite_image = Image.new('RGB', (composite_width, composite_height))  # Create empty image of correct side.
    for idx, image in enumerate(images.images):
        row, col = divmod(idx, num_images_per_row)
        composite_image.paste(image, (col * width, row * height))  # Fill it full of the images we genned
    composite_image_bytes = io.BytesIO()  # Turn the image into a file-like object to be saved and or uploaded to discord.
    if SETTINGS["saveinjpg"][0] == "True":
        composite_image.save(composite_image_bytes, format='JPEG')
    else:
        composite_image.save(composite_image_bytes, format='PNG')
    composite_image_bytes.seek(0)  # Return to the beginning of the file object before we return it.
    return composite_image_bytes


class Imagegenbuttons(discord.ui.View):
    """Class for the ui buttons on speakgen"""
    def __init__(self, generation_queue, prompt, channel, sdmodel, batch_size, username, userid, negativeprompt, steps, width, height, metatron_client, use_defaults):
        super().__init__()
        self.timeout = None  # Disables the timeout on the buttons
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
        self.use_defaults = use_defaults

    @discord.ui.button(label='Reroll', emoji="🎲", style=discord.ButtonStyle.grey)
    async def reroll(self, interaction: discord.Interaction, button: discord.ui.Button):
        """Rerolls last reply"""
        if self.userid == interaction.user.id:
            if await self.metatron_client.is_room_in_queue(self.userid):
                await interaction.response.send_message("Rerolling...", ephemeral=True, delete_after=5)
                self.metatron_client.generation_queue_concurrency_list[interaction.user.id] += 1
                await self.generation_queue.put(('imagegenerate', interaction.user.id, self.prompt, interaction.channel, self.sdmodel, self.batch_size, self.username, self.negativeprompt, self.seed, self.steps, self.width, self.height, self.use_defaults))
            else:
                await interaction.response.send_message("Queue limit reached, please wait until your current gen or gens finish")

    @discord.ui.button(label='Mail', emoji="✉", style=discord.ButtonStyle.grey)
    async def dmimage(self, interaction: discord.Interaction, button: discord.ui.Button):
        """DMs sound"""
        await interaction.response.send_message("DM'ing image...", ephemeral=True, delete_after=5)
        sound_bytes = await interaction.message.attachments[0].read()
        dm_channel = await interaction.user.create_dm()
        truncated_filename = self.prompt[:1000]
        await dm_channel.send(file=discord.File(io.BytesIO(sound_bytes), filename=f'{truncated_filename}.png'))
        speak_dm_logger = logger.bind(user=interaction.user.name, userid=interaction.user.id)
        speak_dm_logger.success("IMAGEGEN DM successful")

    @discord.ui.button(label='Delete', emoji="❌", style=discord.ButtonStyle.grey)
    async def delete_message(self, interaction: discord.Interaction, button: discord.ui.Button):
        """Deletes message"""
        if self.userid == interaction.user.id:
            await interaction.message.delete()
        await interaction.response.send_message("Image deleted.", ephemeral=True, delete_after=5)
        speak_delete_logger = logger.bind(user=interaction.user.name, userid=interaction.user.id)
        speak_delete_logger.info("IMAGEGEN Delete")
