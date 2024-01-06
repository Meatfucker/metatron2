# imagegen_xl.py - Functions for stable diffusion xl capabilities
import asyncio
import io
import math
import os
import re
import random
import gc
import time
import json
from datetime import datetime

import diffusers.utils.logging
import torch
from loguru import logger
from PIL import Image
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline, LCMScheduler
import discord
import jsonlines
from modules.settings import SETTINGS


diffusers.utils.logging.set_verbosity_error()


@logger.catch()
async def load_sdxl_models_list():
    """Get list of models for user interface"""
    models = []
    models_dir = "models/sd-xl/"
    with os.scandir(models_dir) as models_list:
        for models_file in models_list:
            if models_file.name.endswith(".safetensors"):
                models.append(models_file.name)
    return models

@logger.catch()
async def load_sdxl_refiners_list():
    """Get list of models for user interface"""
    models = []
    models_dir = "models/sd-xl-refiners/"
    with os.scandir(models_dir) as models_list:
        for models_file in models_list:
            if models_file.name.endswith(".safetensors"):
                models.append(models_file.name)
    return models

@logger.catch()
async def load_sdxl_loras_list():
    """Get list of Loras for user interface"""
    loras = []
    loras_dir = "models/sd-xl-loras/"
    with os.scandir(loras_dir) as loras_list:
        for loras_file in loras_list:
            if loras_file.name.endswith(".safetensors"):
                token_name = f"{loras_file.name[:-12]}"
                loras.append(token_name)
    return loras

@logger.catch()
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

def jiggle_prompt(search_string):
    found_words = {}
    updated_string = search_string.split()  # Create a list to store updated words
    with jsonlines.open('modules/thesaurus.jsonl') as reader:
        for line in reader:
            for i, word in enumerate(updated_string):
                if len(word) >= 3:  # Check if word is 3 letters or longer
                    if 'word' in line and word == line['word']:
                        if 'synonyms' in line and line['synonyms'] and isinstance(line['synonyms'], list):
                            if word not in found_words:
                                found_words[word] = random.choice(line['synonyms'])
                            else:
                                existing_synonym = found_words[word]
                                new_synonym = random.choice(line['synonyms'])
                                selected_synonym = random.choice([existing_synonym, new_synonym])
                                found_words[word] = selected_synonym
                                updated_string[i] = selected_synonym

    return ' '.join(updated_string)  # Return the updated string


@logger.catch()
async def get_defaults(idname):
    """ This function takes a filename and returns the defaults in it as a dict"""
    filename = f'defaults/{idname}.cfg'
    defaults = {}
    try:
        with open(filename, "r", encoding="utf-8") as defaults_file:
            for defaults_line in defaults_file:
                if "=" in defaults_line:
                    defaults_key, defaults_value = (defaults_line.split("=", 1)[0].strip(), defaults_line.split("=", 1)[1].strip())
                    if defaults_key in defaults:
                        if isinstance(defaults[defaults_key], list):
                            defaults[defaults_key].append(defaults_value)
                        else:
                            defaults[defaults_key] = [defaults[defaults_key], defaults_value]
                    else:
                        defaults[defaults_key] = [defaults_value]
    except FileNotFoundError:
        return None
    return defaults


class ImageXLQueueObject:
    def __init__(self, action, metatron, user, channel, prompt, prompt_2=None, negative_prompt=None, negative_prompt_2=None, model=None, batch_size=None, seed=None, steps=None, width=None, height=None, use_defaults=True, reroll=False):
        self.action = action  # This is the queue action to do
        self.metatron = metatron  # This is the discord client
        self.user = user  # The discord user variable, contains .name and .id
        self.channel = channel  # The discord channel variable, has a bunch of built in functions like sending messages
        self.prompt = prompt  # The users prompt
        self.prompt_2 = prompt_2
        self.negative_prompt = negative_prompt  # The users negative prompt
        self.negative_prompt_2 = negative_prompt_2
        self.model = model  # The requested model
        self.batch_size = batch_size  # The batch size
        self.seed = seed  # The generation seed
        self.steps = steps  # How many inference steps
        self.width = width  # The gen width
        self.height = height  # The gen height
        self.use_defaults = use_defaults  # If this is set to true, itll use channel defaults, otherwise no. Server defaults are still enforced iirc
        self.image = None  # This holds the resulting image after a generation call
        self.processed_prompt = prompt  # This is the prompt that is sent to the generator after being moderated and having loras removed
        self.processed_prompt_2 = prompt_2
        self.processed_negative_prompt = negative_prompt  # This is the negative prompt the is sent to the generator after being moderated and having loras removed
        self.processed_negative_prompt_2 = negative_prompt_2
        self.sd_defaults = None  # This holds the global defauls
        self.channel_defaults = None  # This holds the channel defaults
        self.generation_time = None  # The generation time in seconds
        self.reroll = reroll

    @logger.catch()
    async def load_sd_xl(self):
        """Load a sd model, returning the pipeline object and the compel processor object for the pipeline"""
        logger.debug("SDXL Model Loading...")
        if self.model is not None:  # This loads a checkpoint file
            with torch.no_grad():  # clear gpu memory cache
                self.metatron.sd_xl_pipeline = None
                self.metatron.sd_xl_compel_processor = None
                torch.cuda.empty_cache()
                gc.collect()

                model_id = f'./models/sd-xl/{self.model}'
                self.metatron.sd_xl_pipeline = await asyncio.to_thread(StableDiffusionXLPipeline.from_single_file, model_id, load_safety_checker=False, torch_dtype=torch.float16, use_safetensors=True, custom_pipeline="lpw_stable_diffusion_xl")

        else:
            sd_model_list = await load_sdxl_models_list()
            if sd_model_list is not None:   # This loads a checkpoint file
                with torch.no_grad():  # clear gpu memory cache
                    self.metatron.sd_xl_pipeline = None
                    self.metatron.sd_xl_compel_processor = None
                    torch.cuda.empty_cache()
                    gc.collect()
                    model_id = f'./models/sd-xl/{sd_model_list[0]}'
                    self.metatron.sd_xl_pipeline = await asyncio.to_thread(StableDiffusionXLPipeline.from_single_file, model_id, load_safety_checker=False, torch_dtype=torch.float16, use_safetensors=True, custom_pipeline="lpw_stable_diffusion_xl")
        self.metatron.sd_xl_pipeline.enable_model_cpu_offload()
        self.metatron.sd_xl_loaded_model = self.model


        load_sd_logger = logger.bind(model=model_id)
        load_sd_logger.success("SDXL Model Loaded.")

        with torch.no_grad():  # clear gpu memory cache
            torch.cuda.empty_cache()
        gc.collect()  # clear python memory


    async def load_sdxl_lora(self):
        """ This loads a lora and applies it to a pipeline"""
        with torch.no_grad():  # clear gpu memory cache
            loramatches = re.findall(r'<lora:([^:]+):([\d.]+)>', self.prompt)
            if loramatches:
                names = []
                weights = []
                logger.debug("LORA Loading...")
                for match in loramatches:
                    loraname, loraweight = match
                    loraweight = float(loraweight)  # Convert to a float if needed
                    lorafilename = f'{loraname}.safetensors'
                    self.metatron.sd_xl_pipeline.load_lora_weights("./models/sd-xl-loras", weight_name=lorafilename, adapter_name=loraname)
                    names.append(loraname)
                    weights.append(loraweight)
                self.metatron.sd_xl_pipeline.set_adapters(names, adapter_weights=weights)
                self.processed_prompt = re.sub(r'<lora:([^\s:]+):([\d.]+)>', '', self.prompt)  # This removes the lora trigger from the users prompt so it doesnt effect the gen.

                torch.cuda.empty_cache()
                gc.collect()  # clear python memory

    @logger.catch()
    @torch.no_grad()
    async def generate(self):
        """this generates the request, tiles the images, and returns them as a single image"""

        await self.enforce_defaults_and_limits()
        await self.load_request_or_default_model()
        await self.load_sdxl_lora()
        inputs = await self.get_inputs()  # This creates the prompt embeds
        self.metatron.sd_xl_pipeline.set_progress_bar_config(disable=True)  # This disables the annoying progress bar.
        sd_generate_logger = logger.bind(prompt=self.prompt, prompt_2=self.prompt_2, negative_prompt=self.negative_prompt, negative_prompt_2=self.negative_prompt_2, model=self.model)
        sd_generate_logger.info("IMAGEGEN Generate Started.")
        with torch.no_grad():  # do the generate in a thread so as not to lock up the bot client, and no_grad to save memory.
            start_time = time.time()
            images = await asyncio.to_thread(self.metatron.sd_xl_pipeline, **inputs, num_inference_steps=self.steps, width=self.width, height=self.height)  # do the generate in a thread so as not to lock up the bot client
            end_time = time.time()
            self.generation_time = "{:.3f}".format(end_time - start_time)
            self.metatron.sd_xl_pipeline.unload_lora_weights()
        sd_generate_logger.debug("IMAGEGEN Generate Finished.")
        self.image = await make_image_grid(images)  # Turn returned images into a single image

    @logger.catch()
    async def load_request_or_default_model(self):
        if self.model is not None:  # if a model has been selected, create and load a fresh pipeline and compel processor
            if self.metatron.sd_xl_loaded_model != self.model:  # Only load the model if we dont already have it loaded
                with torch.no_grad():  # clear gpu memory cache
                    torch.cuda.empty_cache()
                    await self.load_sd_xl()
        else:
            if self.metatron.sd_xl_loaded_model != self.sd_defaults["sdxlmodel"][0]:  # If the current model isnt the default model, load it.
                self.model = self.sd_defaults["sdxlmodel"][0]
                with torch.no_grad():  # clear gpu memory cache
                    torch.cuda.empty_cache()
                    await self.load_sd_xl()
            gc.collect()  # clear python memory

    @logger.catch()
    async def enforce_defaults_and_limits(self):
        """The enforces the defaults and max limits"""
        self.sd_defaults = await get_defaults('global')
        self.channel_defaults = await get_defaults(self.channel.id)
        if self.use_defaults is True:
            if self.channel_defaults is not None:
                self.sd_defaults = self.channel_defaults
        if self.batch_size is None:  # Enforce various defaults
            self.batch_size = int(self.sd_defaults["sdxlbatchsize"][0])
        if self.steps is None:
            self.steps = int(self.sd_defaults["sdxlsteps"][0])
        if self.width is None:
            self.width = int(self.sd_defaults["sdxlwidth"][0])
        if self.height is None:
            self.height = int(self.sd_defaults["sdxlheight"][0])
        if self.batch_size > int(SETTINGS["sdxlmaxbatch"][0]):  # These lines ensure compliance with the maxbatch and maxres settings.
            self.batch_size = int(SETTINGS["sdxlmaxbatch"][0])
        if self.width > int(SETTINGS["sdxlmaxres"][0]):
            self.width = int(SETTINGS["sdxlmaxres"][0])
        if self.height > int(SETTINGS["sdxlmaxres"][0]):
            self.height = int(SETTINGS["sdxlmaxres"][0])
        self.width = math.ceil(self.width / 8) * 8  # Dimensions have to be multiple of 8 or else SD shits itself.
        self.height = math.ceil(self.height / 8) * 8
        if self.prompt is not None and self.reroll is not True:
            if self.sd_defaults["imageprompt"][0] not in self.prompt:  # Combine the defaults with the users prompt and negative prompt.
                self.prompt = f'{self.sd_defaults["imageprompt"][0]} {self.prompt}'
        if self.reroll is True:
            self.reroll = False
        if self.prompt_2 is not None:
            if self.sd_defaults["imageprompt"][0] not in self.prompt_2:  # Combine the defaults with the users prompt and negative prompt.
                self.prompt_2 = f'{self.sd_defaults["imageprompt"][0]} {self.prompt_2}'
        if self.negative_prompt is not None:
            if self.sd_defaults["imagenegprompt"][0] not in self.processed_negative_prompt:
                self.processed_negative_prompt = f'{self.sd_defaults["imagenegprompt"][0]} {self.processed_negative_prompt}'
        else:
            self.processed_negative_prompt = self.sd_defaults["imagenegprompt"][0]
        if self.negative_prompt_2 is not None:
            if self.sd_defaults["imagenegprompt"][0] not in self.processed_negative_prompt_2:
                self.processed_negative_prompt_2 = f'{self.sd_defaults["imagenegprompt"][0]} {self.processed_negative_prompt_2}'
        else:
            self.processed_negative_prompt_2 = self.sd_defaults["imagenegprompt"][0]


    @logger.catch()
    async def get_inputs(self):
        """This multiples the prompt by the batch size and creates the weight embeddings"""
        if self.seed is None:  # Use a random seed if one isnt supplied
            generator = [torch.Generator("cuda").manual_seed(random.randint(-2147483648, 2147483647)) for _ in range(self.batch_size)]
        else:
            generator = [torch.Generator("cuda").manual_seed(self.seed + i) for i in range(self.batch_size)]
        await self.moderate_prompt()  # Moderate prompt according to settings.
        prompts = self.batch_size * [self.processed_prompt]
        inputs_dict = {}
        inputs_dict.update({"prompt": prompts})
        if self.processed_prompt_2 is not None:
            prompts_2 = self.batch_size * [self.processed_prompt_2]
            inputs_dict.update({"prompt_2": prompts_2})
        if self.processed_negative_prompt is not None:
            negativeprompts = self.batch_size * [self.processed_negative_prompt]
            inputs_dict.update({"negative_prompt": negativeprompts})
        if self.processed_negative_prompt_2 is not None:
            negativeprompts_2 = self.batch_size * [self.processed_negative_prompt_2]
            inputs_dict.update({"negative_prompt_2": negativeprompts_2})

        return inputs_dict


    @logger.catch()
    async def moderate_prompt(self):
        """Removes all words in the imagebannedwords from prompt, adds all words in imagenegprompt to negativeprompt"""
        banned_words = self.sd_defaults["imagebannedwords"][0].split(',')
        for word in banned_words:
            self.processed_prompt = self.processed_prompt.replace(word.strip(), '')
            if self.processed_prompt_2 is not None:
                self.processed_prompt_2 = self.processed_prompt_2.replace(word.strip(), '')
        imagenegprompt_string = self.sd_defaults["imagenegprompt"][0]
        if self.negative_prompt is None:
            self.processed_negative_prompt = ""
        self.processed_negative_prompt = self.processed_negative_prompt + " " + imagenegprompt_string
        if self.negative_prompt_2 is None:
            self.processed_negative_prompt_2 = ""
        self.processed_negative_prompt_2 = self.processed_negative_prompt_2 + " " + imagenegprompt_string



    @logger.catch()
    async def save(self):
        """Saves image to disk"""
        sanitized_prompt = re.sub(r'[^\w\s\-.]', '', self.processed_prompt)[:100]
        current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if SETTINGS["saveinjpg"][0] == "True":
            savepath = f'{SETTINGS["savepath"][0]}/{current_datetime}-{sanitized_prompt}.jpg'
        else:
            savepath = f'{SETTINGS["savepath"][0]}/{current_datetime}-{sanitized_prompt}.png'
        with open(savepath, "wb") as output_file:
            output_file.write(self.image.getvalue())

    @logger.catch()
    async def respond(self):
        sanitized_prompt = re.sub(r'[^\w\s\-.]', '', self.processed_prompt)[:100]
        if SETTINGS["saveinjpg"][0] == "True":  # Save and upload in jpg if enabled, otherwise PNG
            await self.channel.send(content=f"User: {self.user.mention} Prompt:`{self.prompt}` Prompt_2:`{self.prompt_2}` Negative:`{self.negative_prompt}` Model:`{self.model}` Batch Size:`{self.batch_size}` Seed:`{self.seed}` Steps:`{self.steps}` Width:`{self.width}` Height:`{self.height}` Time:`{self.generation_time} seconds`", file=discord.File(self.image, filename=f"{sanitized_prompt}.jpg"), view=Imagegenbuttons(self))
        else:
            await self.channel.send(content=f"User: {self.user.mention} Prompt:`{self.prompt}` Prompt_2:`{self.prompt_2}` Negative:`{self.negative_prompt}` Model:`{self.model}` Batch Size:`{self.batch_size}` Seed:`{self.seed}` Steps:`{self.steps}` Width:`{self.width}` Height:`{self.height}` Time:`{self.generation_time} seconds`", file=discord.File(self.image, filename=f"{sanitized_prompt}.png"), view=Imagegenbuttons(self))
        imagegenreply_logger = logger.bind(user=self.user.name, prompt=self.prompt, prompt_2=self.prompt_2, negativeprompt=self.negative_prompt, model=self.model)
        imagegenreply_logger.success("IMAGEGEN Replied")


class Imagegenbuttons(discord.ui.View):
    """Class for the ui buttons on speakgen"""
    def __init__(self, imageobject):
        super().__init__()
        self.timeout = None  # Disables the timeout on the buttons
        self.imageobject = imageobject

    @discord.ui.button(label='Reroll', emoji="🎲", style=discord.ButtonStyle.grey)
    async def reroll(self, interaction: discord.Interaction, button: discord.ui.Button):
        """Rerolls last reply"""
        if self.imageobject.user.id == interaction.user.id:
            if await self.imageobject.metatron.is_room_in_queue(self.imageobject.user.id):
                await interaction.response.send_message("Rerolling...", ephemeral=True, delete_after=5)
                self.imageobject.metatron.generation_queue_concurrency_list[interaction.user.id] += 1
                await self.imageobject.metatron.generation_queue.put(self.imageobject)
            else:
                await interaction.response.send_message("Queue limit reached, please wait until your current gen or gens finish")

    @discord.ui.button(label='Jiggle', emoji="🔀", style=discord.ButtonStyle.grey)
    async def jiggle(self, interaction: discord.Interaction, button: discord.ui.Button):
        """Jiggle last reply"""
        if self.imageobject.user.id == interaction.user.id:
            self.imageobject.prompt = jiggle_prompt(self.imageobject.prompt)
            self.imageobject.reroll = True
            if await self.imageobject.metatron.is_room_in_queue(self.imageobject.user.id):
                await interaction.response.send_message("Jiggling...", ephemeral=True, delete_after=5)
                self.imageobject.metatron.generation_queue_concurrency_list[interaction.user.id] += 1
                await self.imageobject.metatron.generation_queue.put(self.imageobject)
            else:
                await interaction.response.send_message("Queue limit reached, please wait until your current gen or gens finish")

    @discord.ui.button(label='Mail', emoji="✉", style=discord.ButtonStyle.grey)
    async def dmimage(self, interaction: discord.Interaction, button: discord.ui.Button):
        """DMs sound"""
        await interaction.response.send_message("DM'ing image...", ephemeral=True, delete_after=5)
        image_bytes = await interaction.message.attachments[0].read()
        dm_channel = await interaction.user.create_dm()
        sanitized_prompt = re.sub(r'[^\w\s\-.]', '', self.imageobject.processed_prompt)[:100]
        await dm_channel.send(file=discord.File(io.BytesIO(image_bytes), filename=f'{sanitized_prompt}.png'))
        image_dm_logger = logger.bind(user=interaction.user.name, userid=interaction.user.id)
        image_dm_logger.success("IMAGEGEN DM successful")

    @discord.ui.button(label='Delete', emoji="❌", style=discord.ButtonStyle.grey)
    async def delete_message(self, interaction: discord.Interaction, button: discord.ui.Button):
        """Deletes message"""
        if self.imageobject.user.id == interaction.user.id:
            await interaction.message.delete()
        await interaction.response.send_message("Image deleted.", ephemeral=True, delete_after=5)
        speak_delete_logger = logger.bind(user=interaction.user.name, userid=interaction.user.id)
        speak_delete_logger.info("IMAGEGEN Delete")
