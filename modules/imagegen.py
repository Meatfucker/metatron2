# imagegen.py - Functions for stable diffusion capabilities
import asyncio
import io
import math
import os
import re
import random
import gc
from datetime import datetime

import diffusers.utils.logging
import torch
from loguru import logger
from PIL import Image
from diffusers import DiffusionPipeline, StableDiffusionPipeline, DPMSolverSinglestepScheduler, DDIMScheduler
from compel import Compel
import discord
from modules.settings import SETTINGS


diffusers.utils.logging.set_verbosity_error()


async def load_models_list():
    """Get list of models for user interface"""
    models = []
    models_dir = "models/sd/"
    with os.scandir(models_dir) as models_list:
        for models_file in models_list:
            if models_file.name.endswith(".safetensors"):
                models.append(models_file.name)
    return models


async def load_loras_list():
    """Get list of Loras for user interface"""
    loras = []
    loras_dir = "models/sd-loras/"
    with os.scandir(loras_dir) as loras_list:
        for loras_file in loras_list:
            if loras_file.name.endswith(".safetensors"):
                token_name = f"{loras_file.name[:-12]}"
                loras.append(token_name)
    return loras


async def load_embeddings_list():
    """Get list of embeddings for user interface"""
    embeddings = []
    embeddings_dir = "models/sd-embeddings/"
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
        model_id = f'./models/sd/{model}'
        pipeline = await asyncio.to_thread(StableDiffusionPipeline.from_single_file, model_id, load_safety_checker=False, torch_dtype=torch.float16, use_safetensors=True, custom_pipeline="stable_diffusion_tensorrt_txt2img", revision='fp16')
    else:
        sd_model_list = await load_models_list()
        if sd_model_list is not None:   # This loads a checkpoint file
            model_id = f'./models/sd/{sd_model_list[0]}'
            pipeline = await asyncio.to_thread(StableDiffusionPipeline.from_single_file, model_id, load_safety_checker=False, torch_dtype=torch.float16, use_safetensors=True, custom_pipeline="stable_diffusion_tensorrt_txt2img", revision='fp16')
        else:  # This loads a huggingface based model, is a fallback for if there are no models
            model_id = "runwayml/stable-diffusion-v1-5"
            pipeline = await asyncio.to_thread(DiffusionPipeline.from_pretrained, model_id, safety_checker=None, torch_dtype=torch.float16, use_safetensors=True)
    pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)  # This is the sampler, I may make it configurable in the future
    pipeline = pipeline.to("cuda")  # push the pipeline to gpu
    load_sd_logger = logger.bind(model=model_id)
    load_sd_logger.success("SD Model Loaded.")
    compel_proc = Compel(tokenizer=pipeline.tokenizer, text_encoder=pipeline.text_encoder, truncate_long_prompts=False)  # create the compel processor object for the pipeline
    with torch.no_grad():  # clear gpu memory cache
        torch.cuda.empty_cache()
    gc.collect()  # clear python memory
    return pipeline, compel_proc, model


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


class ImageQueueObject:
    def __init__(self, action, metatron, user, channel, prompt, negative_prompt=None, model=None, batch_size=None, seed=None, steps=None, width=None, height=None, use_defaults=True):
        self.action = action  # This is the queue action to do
        self.metatron = metatron  # This is the discord client
        self.user = user  # The discord user variable, contains .name and .id
        self.channel = channel  # The discord channel variable, has a bunch of built in functions like sending messages
        self.prompt = prompt  # The users prompt
        self.negative_prompt = negative_prompt  # The users negative prompt
        self.model = model  # The requested model
        self.batch_size = batch_size  # The batch size
        self.seed = seed  # The generation seed
        self.steps = steps  # How many inference steps
        self.width = width  # The gen width
        self.height = height  # The gen height
        self.use_defaults = use_defaults  # If this is set to true, itll use channel defaults, otherwise no. Server defaults are still enforced iirc
        self.image = None  # This holds the resulting image after a generation call
        self.processed_prompt = prompt  # This is the prompt that is sent to the generator after being moderated and having loras removed
        self.processed_negative_prompt = negative_prompt  # This is the negative prompt the is sent to the generator after being moderated and having loras removed
        self.sd_defaults = None  # This holds the global defauls
        self.channel_defaults = None  # This holds the channel defaults

    @torch.no_grad()
    async def generate(self):
        """this generates the request, tiles the images, and returns them as a single image"""

        await self.enforce_defaults_and_limits()
        await self.load_request_or_default_model()
        await self.load_sd_lora()  # Check the prompt for loras and load them if needed.
        await self.load_ti()  # Check the prompt for TIs and load them if needed.
        inputs = await self.get_inputs()  # This creates the prompt embeds
        self.metatron.sd_pipeline.set_progress_bar_config(disable=True)  # This disables the annoying progress bar.
        sd_generate_logger = logger.bind(prompt=self.prompt, negative_prompt=self.negative_prompt, model=self.model)
        sd_generate_logger.info("IMAGEGEN Generate Started.")
        with torch.no_grad():  # do the generate in a thread so as not to lock up the bot client, and no_grad to save memory.
            images = await asyncio.to_thread(self.metatron.sd_pipeline, **inputs, num_inference_steps=self.steps, width=self.width, height=self.height)  # do the generate in a thread so as not to lock up the bot client
            self.metatron.sd_pipeline.unload_lora_weights()  # Unload the loras so they dont effect future gens if they dont change models.
        sd_generate_logger.debug("IMAGEGEN Generate Finished.")
        self.image = await make_image_grid(images)  # Turn returned images into a single image

    async def load_request_or_default_model(self):
        if self.model is not None:  # if a model has been selected, create and load a fresh pipeline and compel processor
            if self.metatron.sd_loaded_model != self.model:  # Only load the model if we dont already have it loaded
                self.metatron.sd_pipeline, self.metatron.sd_compel_processor, self.metatron.sd_loaded_model = await load_sd(self.model)
                self.metatron.sd_loaded_embeddings = []  # Since we loaded a new model, clear the loaded embeddings list
                with torch.no_grad():  # clear gpu memory cache
                    torch.cuda.empty_cache()
                gc.collect()  # clear python memory
        else:
            if self.metatron.sd_loaded_model != self.sd_defaults["imagemodel"][0]:  # If the current model isnt the default model, load it.
                self.metatron.sd_pipeline, self.metatron.sd_compel_processor, self.metatron.sd_loaded_model = await load_sd(self.sd_defaults["imagemodel"][0])
                self.metatron.sd_loaded_embeddings = []  # Since we loaded a new model, clear the loaded embeddings list
            with torch.no_grad():  # clear gpu memory cache
                torch.cuda.empty_cache()
            gc.collect()  # clear python memory

    async def enforce_defaults_and_limits(self):
        """The enforces the defaults and max limits"""
        self.sd_defaults = await get_defaults('global')
        self.channel_defaults = await get_defaults(self.channel.id)
        if self.use_defaults is True:
            if self.channel_defaults is not None:
                self.sd_defaults = self.channel_defaults
        if self.batch_size is None:  # Enforce various defaults
            self.batch_size = int(self.sd_defaults["imagebatchsize"][0])
        if self.steps is None:
            self.steps = int(self.sd_defaults["imagesteps"][0])
        if self.width is None:
            self.width = int(self.sd_defaults["imagewidth"][0])
        if self.height is None:
            self.height = int(self.sd_defaults["imageheight"][0])
        if self.batch_size > int(SETTINGS["maxbatch"][0]):  # These lines ensure compliance with the maxbatch and maxres settings.
            self.batch_size = int(SETTINGS["maxbatch"][0])
        if self.width > int(SETTINGS["maxres"][0]):
            self.width = int(SETTINGS["maxres"][0])
        if self.height > int(SETTINGS["maxres"][0]):
            self.height = int(SETTINGS["maxres"][0])
        self.width = math.ceil(self.width / 8) * 8  # Dimensions have to be multiple of 8 or else SD shits itself.
        self.height = math.ceil(self.height / 8) * 8
        if self.prompt is not None:
            if self.sd_defaults["imageprompt"][0] not in self.prompt:  # Combine the defaults with the users prompt and negative prompt.
                self.prompt = f'{self.sd_defaults["imageprompt"][0]} {self.prompt}'
        if self.negative_prompt is not None:
            if self.sd_defaults["imagenegprompt"][0] not in self.processed_negative_prompt:
                self.processed_negative_prompt = f'{self.sd_defaults["imagenegprompt"][0]} {self.processed_negative_prompt}'
        else:
            self.processed_negative_prompt = self.sd_defaults["imagenegprompt"][0]

    async def get_inputs(self):
        """This multiples the prompt by the batch size and creates the weight embeddings"""
        if self.seed is None:  # Use a random seed if one isnt supplied
            generator = [torch.Generator("cuda").manual_seed(random.randint(-2147483648, 2147483647)) for _ in range(self.batch_size)]
        else:
            generator = [torch.Generator("cuda").manual_seed(self.seed + i) for i in range(self.batch_size)]
        await self.moderate_prompt()  # Moderate prompt according to settings.
        await self.format_prompt_weights()  # Apply compel weights
        prompts = self.batch_size * [self.processed_prompt]
        with torch.no_grad():
            prompt_embeds = self.metatron.sd_compel_processor(prompts)  # Make compel embeddings
        if self.processed_negative_prompt is not None:
            negativeprompts = self.batch_size * [self.processed_negative_prompt]
            with torch.no_grad():
                negative_prompt_embeds = self.metatron.sd_compel_processor(negativeprompts)
                [prompt_embeds, negative_prompt_embeds] = self.metatron.sd_compel_processor.pad_conditioning_tensors_to_same_length([prompt_embeds, negative_prompt_embeds])
            return {"prompt_embeds": prompt_embeds, "negative_prompt_embeds": negative_prompt_embeds, "generator": generator}
        else:
            return {"prompt_embeds": prompt_embeds, "generator": generator}

    async def format_prompt_weights(self):
        """This takes a prompt, checks for A1111 style prompt weightings, and converts them to compel style weightings"""
        matches = re.findall(r'\((.*?)\:(.*?)\)', self.processed_prompt)
        for match in matches:
            words = match[0].split()
            number = match[1]
            replacement = ' '.join(f"({word}){number}" for word in words)
            self.processed_prompt = self.processed_prompt.replace(f"({match[0]}:{match[1]})", replacement)
        if self.negative_prompt is not None:
            matches = re.findall(r'\((.*?)\:(.*?)\)', self.processed_negative_prompt)
            for match in matches:
                words = match[0].split()
                number = match[1]
                replacement = ' '.join(f"({word}){number}" for word in words)
                self.processed_negative_prompt = self.processed_negative_prompt.replace(f"({match[0]}:{match[1]})", replacement)

    async def moderate_prompt(self):
        """Removes all words in the imagebannedwords from prompt, adds all words in imagenegprompt to negativeprompt"""
        banned_words = self.sd_defaults["imagebannedwords"][0].split(',')
        for word in banned_words:
            self.processed_prompt = self.processed_prompt.replace(word.strip(), '')
        imagenegprompt_string = self.sd_defaults["imagenegprompt"][0]
        if self.negative_prompt is None:
            self.processed_negative_prompt = ""
        self.processed_negative_prompt = self.processed_negative_prompt + " " + imagenegprompt_string

    async def load_sd_lora(self):
        """ This loads a lora and applies it to a pipeline"""
        loramatches = re.findall(r'<lora:([^:]+):([\d.]+)>', self.prompt)
        if loramatches:
            names = []
            weights = []
            logger.debug("LORA Loading...")
            for match in loramatches:
                loraname, loraweight = match
                loraweight = float(loraweight)  # Convert to a float if needed
                lorafilename = f'{loraname}.safetensors'
                self.metatron.sd_pipeline.load_lora_weights("./models/sd-loras", weight_name=lorafilename, adapter_name=loraname)
                names.append(loraname)
                weights.append(loraweight)
            self.metatron.sd_pipeline.set_adapters(names, adapter_weights=weights)
            self.processed_prompt = re.sub(r'<lora:([^\s:]+):([\d.]+)>', '', self.prompt)  # This removes the lora trigger from the users prompt so it doesnt effect the gen.
            with torch.no_grad():  # clear gpu memory cache
                torch.cuda.empty_cache()
            gc.collect()  # clear python memory

    async def load_ti(self):
        """this checks the prompt for textual inversion embeddings and loads them if needed, keeping track of used ones so it doesnt try to load the same one twice"""
        matches = re.findall(r'<(.*?)>', self.processed_prompt)
        for match in matches:
            file_path = f".models/sd-embeddings/{match}.pt"
            token_name = f"<{match}>"
            if match not in self.metatron.sd_loaded_embeddings:
                if os.path.exists(file_path):
                    self.metatron.sd_pipeline.load_textual_inversion(file_path, token=token_name)  # this applies the embedding to the pipeline
                    self.metatron.sd_loaded_embeddings.append(match)  # add the embedding to the current loaded list

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

    async def respond(self):
        sanitized_prompt = re.sub(r'[^\w\s\-.]', '', self.processed_prompt)[:100]
        if SETTINGS["saveinjpg"][0] == "True":  # Save and upload in jpg if enabled, otherwise PNG
            await self.channel.send(content=f"User: `{self.user.name}` Prompt:`{self.prompt}` Negative:`{self.negative_prompt}` Model:`{self.model}` Batch Size:`{self.batch_size}` Seed:`{self.seed}` Steps:`{self.steps}` Width:`{self.width}` Height:`{self.height}` ", file=discord.File(self.image, filename=f"{sanitized_prompt}.jpg"), view=Imagegenbuttons(self))
        else:
            await self.channel.send(content=f"User: `{self.user.name}` Prompt:`{self.prompt}` Negative:`{self.negative_prompt}` Model:`{self.model}` Batch Size:`{self.batch_size}` Seed:`{self.seed}` Steps:`{self.steps}` Width:`{self.width}` Height:`{self.height}` ", file=discord.File(self.image, filename=f"{sanitized_prompt}.png"), view=Imagegenbuttons(self))
        imagegenreply_logger = logger.bind(user=self.user.name, prompt=self.prompt, negativeprompt=self.negative_prompt, model=self.model)
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
