
import json
import base64
import os
import math
from loguru import logger
from datetime import datetime
import discord
import aiohttp
import re
import io
import time
from PIL import Image
import random
from modules.settings import SETTINGS


async def api_load_sd_models():
    """Get list of models for user interface"""
    models = []
    async with aiohttp.ClientSession() as session:
        async with session.get(f'http://{SETTINGS["imageapi"][0]}/sdapi/v1/sd-models') as response:
            response_data = await response.json()
            for title in response_data:
                models.append(title["title"])
    return models


async def api_load_sd_loras():
    """Get list of loras for user interface"""
    loras = []
    async with aiohttp.ClientSession() as session:
        async with session.get(f'http://{SETTINGS["imageapi"][0]}/sdapi/v1/loras') as response:
            response_data = await response.json()
            for name in response_data:
                loras.append(name["name"])
    return loras


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


@logger.catch()
async def make_image_grid(images):
    """This takes a list of pil image objects and turns them into a grid"""
    width, height = images[0].size  # Figure out final image dimensions.
    num_images_per_row = math.ceil(math.sqrt(len(images)))
    num_rows = math.ceil(len(images) / num_images_per_row)
    composite_width = num_images_per_row * width
    composite_height = num_rows * height
    composite_image = Image.new('RGB', (composite_width, composite_height))  # Create empty image of correct side.
    for idx, image in enumerate(images):
        row, col = divmod(idx, num_images_per_row)
        composite_image.paste(image, (col * width, row * height))  # Fill it full of the images we genned
    composite_image_bytes = io.BytesIO()  # Turn the image into a file-like object to be saved and or uploaded to discord.
    if SETTINGS["saveinjpg"][0] == "True":
        composite_image.save(composite_image_bytes, format='JPEG')
    else:
        composite_image.save(composite_image_bytes, format='PNG')
    composite_image_bytes.seek(0)  # Return to the beginning of the file object before we return it.
    return composite_image_bytes


class ApiImageQueueObject:
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
        self.payload = {}
        self.generation_time  # the generation time in seconds.

    async def generate(self):
        """this generates the request, tiles the images, and returns them as a single image"""
        await self.enforce_defaults_and_limits()
        await self.moderate_prompt()
        await self.build_payload()

        sd_generate_logger = logger.bind(prompt=self.prompt, negative_prompt=self.negative_prompt, model=self.model)
        sd_generate_logger.info("IMAGEGEN Generate Started.")
        start_time = time.time()
        async with aiohttp.ClientSession() as session:
            async with session.post(f'http://{SETTINGS["imageapi"][0]}/sdapi/v1/txt2img', json=self.payload) as response:
                if response.status == 200:
                    data = await response.json()
                    if "images" in data:  # Tile and compile images into a grid
                        image_list = [Image.open(io.BytesIO(base64.b64decode(i.split(",", 1)[0]))) for i in data['images']]
                        self.image = await make_image_grid(image_list)  # Turn returned images into a single image
        end_time = time.time()
        self.generation_time = "{:.3f}".format(end_time - start_time)
        sd_generate_logger.debug("IMAGEGEN Generate Finished.")

    async def build_payload(self):
        self.payload.update({"prompt": self.processed_prompt})
        self.payload.update({"negative_prompt": self.processed_negative_prompt})
        self.payload.update({"batch_size": self.batch_size})
        self.payload.update({"seed": self.seed})
        self.payload.update({"steps": self.steps})
        self.payload.update({"width": self.width})
        self.payload.update({"height": self.height})

    @logger.catch()
    async def load_model(self):
        model_payload = {"sd_model_checkpoint": self.model}
        async with aiohttp.ClientSession() as session:  # make the api request to change to the requested model
            async with session.post(f'http://{SETTINGS["imageapi"][0]}/sdapi/v1/options', json=model_payload) as response:
                response_data = await response.json()
                logger.bind(sd_api_model=response_data)
                logger.debug("SD API MODEL")
        self.metatron.sd_loaded_model = self.model

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
        if self.model is None:
            self.model = self.sd_defaults["imagemodel"][0]
            await self.load_model()
        else:
            await self.load_model()

    async def moderate_prompt(self):
        """Removes all words in the imagebannedwords from prompt, adds all words in imagenegprompt to negativeprompt"""
        banned_words = self.sd_defaults["imagebannedwords"][0].split(',')
        for word in banned_words:
            self.processed_prompt = self.processed_prompt.replace(word.strip(), '')
        imagenegprompt_string = self.sd_defaults["imagenegprompt"][0]
        if self.negative_prompt is None:
            self.processed_negative_prompt = ""
        self.processed_negative_prompt = self.processed_negative_prompt + " " + imagenegprompt_string

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
            await self.channel.send(content=f"User: `{self.user.name}` Prompt:`{self.prompt}` Negative:`{self.negative_prompt}` Model:`{self.model}` Batch Size:`{self.batch_size}` Seed:`{self.seed}` Steps:`{self.steps}` Width:`{self.width}` Height:`{self.height}` Time:`{self.generation_time} seconds`", file=discord.File(self.image, filename=f"{sanitized_prompt}.jpg"), view=Imagegenbuttons(self))
        else:
            await self.channel.send(content=f"User: `{self.user.name}` Prompt:`{self.prompt}` Negative:`{self.negative_prompt}` Model:`{self.model}` Batch Size:`{self.batch_size}` Seed:`{self.seed}` Steps:`{self.steps}` Width:`{self.width}` Height:`{self.height}` Time:`{self.generation_time} seconds`", file=discord.File(self.image, filename=f"{sanitized_prompt}.png"), view=Imagegenbuttons(self))
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


class ApiWordQueueObject:

    def __init__(self, action, metatron, user, channel, prompt=None, negative_prompt=None, llm_prompt=None, reroll=False):
        self.action = action  # This is the queue generation action
        self.metatron = metatron  # This is the discord client
        self.user = user  # This is the discord user variable, contains user.name and user.id
        self.channel = channel  # This is the discord channel variable/
        self.prompt = prompt  # This holds the users prompt
        self.payload_prompt = None
        self.negative_prompt = negative_prompt  # This holds the users negative prompt
        self.llm_prompt = llm_prompt  # This holds the llm prompt for /impersonate
        self.llm_response = None  # This holds the resulting response from generate
        self.reroll = reroll  # If this is true, when it generates text itll delete the last q/a pair and replace it with the new one.
        self.payload = {}

    @logger.catch()
    async def build_payload(self):
        payload_defaults = await get_defaults('wordapi')
        self.payload = {key: values[0] for key, values in payload_defaults.items()}
        self.payload.update({"prompt": self.payload_prompt})
        # self.payload.update({"negative_prompt": self.negative_prompt})

    @logger.catch()
    async def generate(self):
        """function for generating responses with the llm"""
        llm_defaults = await get_defaults('global')
        userhistory = await self.load_history()  # load the users past history to include in the prompt
        if self.reroll is True:
            await self.delete_last_history_pair()
            self.reroll = False
        if self.user.id not in self.metatron.llm_user_history or not self.metatron.llm_user_history[self.user.id]:
            self.payload_prompt = f'{llm_defaults["wordsystemprompt"][0]}\n\nUSER:{self.prompt}\nASSISTANT:'  # if there is no history, add the system prompt to the beginning
        else:
            self.payload_prompt = f'{userhistory}\nUSER:{self.prompt}\nASSISTANT:'
        await self.build_payload()
        llm_generate_logger = logger.bind(user=self.user.name, prompt=self.prompt, negative=self.negative_prompt)
        llm_generate_logger.info("WORDGEN Generate Started.")
        logger.debug(self.payload)
        async with aiohttp.ClientSession() as session:  # make the api request
            async with session.post(f'http://{SETTINGS["wordapi"][0]}/v1/completions', json=self.payload, timeout=None) as response:
                if response.status == 200:
                    result = await response.json()
                    self.llm_response = result["choices"][0]["text"]  # load said reply
                else:
                    self.llm_response = "API FAILURE"
        llm_generate_logger.debug("WORDGEN Generate Completed")
        await self.save_history()  # save the response to the users history

    @logger.catch()
    async def summary(self):
        """function for generating and posting chat summary with the llm"""
        channel_history = [message async for message in self.channel.history(limit=20)]
        compiled_messages = '\n'.join([f'{msg.author}: {msg.content}' for msg in channel_history])
        self.payload_prompt = f'You are an AI assistant that summarizes conversations.\n\nUSER: Here is the conversation to: {compiled_messages}\nASSISTANT:'  # if there is no history, add the system prompt to the beginning
        await self.build_payload()
        llm_summary_logger = logger.bind(user=self.user.name)
        llm_summary_logger.info("WORDGEN Summary Started.")
        async with aiohttp.ClientSession() as session:  # make the api request
            async with session.post(f'http://{SETTINGS["wordapi"][0]}/v1/completions', json=self.payload, timeout=None) as response:
                if response.status == 200:
                    result = await response.json()
                    self.llm_response = result["choices"][0]["text"]  # load said reply
        llm_summary_logger.debug("WORDGEN Summary Completed")
        message_chunks = [self.llm_response[i:i + 1500] for i in range(0, len(self.llm_response), 1500)]  # Post the message
        for message in message_chunks:
            await self.channel.send(message)
        llm_summary_logger = logger.bind(user=self.user.name)
        llm_summary_logger.success("SUMMARY Reply")
        gc.collect()

    @logger.catch()
    async def respond(self):
        """Prints the LLM response to the chat"""
        if self.user.id in self.metatron.llm_view_last_message:  # check if there are an existing set of llm buttons for the user and if so, delete them
            try:
                await self.metatron.llm_view_last_message[self.user.id].delete()
            except discord.NotFound:
                pass

        if self.user.id in self.metatron.llm_chunks_messages:  # If its a reroll, delete the old messages
            for chunk_message in self.metatron.llm_chunks_messages[self.user.id]:
                if self.reroll:
                    try:
                        await chunk_message.delete()
                    except discord.NotFound:
                        pass  # Message not found, might have been deleted already
                    finally:
                        self.reroll = False
            del self.metatron.llm_chunks_messages[self.user.id]
        message_chunks = [self.llm_response[i:i + 1500] for i in range(0, len(self.llm_response), 1500)]  # Send and track the previously sent messages in case we have to delete them for reroll.
        if self.user.id not in self.metatron.llm_chunks_messages:
            self.metatron.llm_chunks_messages[self.user.id] = []
        for chunk in message_chunks:
            chunk_message = await self.channel.send(chunk)
            self.metatron.llm_chunks_messages[self.user.id].append(chunk_message)
        new_message = await self.channel.send(view=Wordgenbuttons(self))  # send the message with the llm buttons
        self.metatron.llm_view_last_message[self.user.id] = new_message  # track the message id of the last set of llm buttons for each user
        llm_reply_logger = logger.bind(user=self.user.name, prompt=self.prompt, negative=self.negative_prompt)
        llm_reply_logger.success("WORDGEN Reply")

    @logger.catch()
    async def load_history(self):
        """loads a users history into a single string and returns it"""
        if self.user.id in self.metatron.llm_user_history and self.metatron.llm_user_history[self.user.id]:
            combined_history = ''.join(self.metatron.llm_user_history[self.user.id])
            return combined_history

    @logger.catch()
    async def delete_last_history_pair(self):
        """Deletes the last question/answer pair from a users history"""
        if self.user.id in self.metatron.llm_user_history:
            self.metatron.llm_user_history[self.user.id].pop()

    @logger.catch()
    async def clear_history(self):
        """deletes a users histroy"""
        if self.user.id in self.metatron.llm_user_history:
            del self.metatron.llm_user_history[self.user.id]

    @logger.catch()
    async def insert_history(self):
        """inserts a question/answer pair into a users history"""
        llm_defaults = await get_defaults('global')
        if self.user.id not in self.metatron.llm_user_history:  # if they have no history, include the system prompt
            self.metatron.llm_user_history[self.user.id] = []
            injectedhistory = f'{llm_defaults["wordsystemprompt"][0]}\n\nUSER:{self.prompt}\nASSISTANT:{self.llm_prompt}</s>\n'
        else:
            injectedhistory = f'USER:{self.prompt}\nASSISTANT:{self.llm_prompt}</s>\n'
        if len(self.metatron.llm_user_history[self.user.id]) >= int(llm_defaults["wordmaxhistory"][0]):  # check if the history has reached 20 items
            del self.metatron.llm_user_history[self.user.id][0]
        self.metatron.llm_user_history[self.user.id].append(injectedhistory)

    @logger.catch()
    async def save_history(self):
        """saves the prompt and llm response to the users history"""
        llm_defaults = await get_defaults('global')
        if self.user.id not in self.metatron.llm_user_history:  # if they have no history yet include the system prompt along with the special tokens for the instruction format
            self.metatron.llm_user_history[self.user.id] = []
            logger.debug(self.llm_response)
            messagepair = f'{llm_defaults["wordsystemprompt"][0]}\n\nUSER:{self.prompt}\nASSISTANT:{self.llm_response}</s>\n'
        else:
            messagepair = f'USER:{self.prompt}\nASSISTANT:{self.llm_response}</s>\n'  # otherwise just the message pair and special tokens
        if len(self.metatron.llm_user_history[self.user.id]) >= int(llm_defaults["wordmaxhistory"][0]):  # check if the history has reached 20 items
            del self.metatron.llm_user_history[self.user.id][0]
        self.metatron.llm_user_history[self.user.id].append(messagepair)  # add the message to the history


class Wordgenbuttons(discord.ui.View):
    """Class for the ui buttons on speakgen"""

    def __init__(self, wordobject):
        super().__init__()
        self.timeout = None  # makes the buttons never time out
        self.wordobject = wordobject

    @discord.ui.button(label='Reroll last reply', emoji="🎲", style=discord.ButtonStyle.grey)
    async def reroll(self, interaction: discord.Interaction, button: discord.ui.Button):
        """Rerolls last reply"""
        if self.wordobject.user.id == interaction.user.id:
            if await self.wordobject.metatron.is_room_in_queue(self.wordobject.user.id):
                await interaction.response.send_message("Rerolling...", ephemeral=True, delete_after=5)
                self.wordobject.metatron.generation_queue_concurrency_list[interaction.user.id] += 1
                self.wordobject.reroll = True
                self.wordobject.action = "wordgen"
                await self.wordobject.metatron.generation_queue.put(self.wordobject)
            else:
                await interaction.response.send_message("Queue limit reached, please wait until your current gen or gens finish")

    @discord.ui.button(label='Delete last reply', emoji="❌", style=discord.ButtonStyle.grey)
    async def delete_message(self, interaction: discord.Interaction, button: discord.ui.Button):
        """Deletes message"""
        if self.wordobject.user.id == interaction.user.id:
            if await self.wordobject.metatron.is_room_in_queue(self.wordobject.user.id):
                self.wordobject.metatron.generation_queue_concurrency_list[interaction.user.id] += 1
                self.wordobject.action = "wordgendeletelast"
                await self.wordobject.metatron.generation_queue.put(self.wordobject)
                await interaction.response.send_message("Last question/answer pair deleted", ephemeral=True, delete_after=5)
            else:
                await interaction.response.send_message("Queue limit reached, please wait until your current gen or gens finish")

    @discord.ui.button(label='Show History', emoji="📜", style=discord.ButtonStyle.grey)
    async def dm_history(self, interaction: discord.Interaction, button: discord.ui.Button):
        """Prints history to user"""
        if self.wordobject.user.id == interaction.user.id:
            if self.wordobject.user.id in self.wordobject.metatron.llm_user_history:
                history_file = io.BytesIO(json.dumps(self.wordobject.metatron.llm_user_history[self.wordobject.user.id], indent=1).encode())
                await interaction.response.send_message(ephemeral=True, file=discord.File(history_file, filename='history.txt'))
            else:
                await interaction.response.send_message("No History", ephemeral=True, delete_after=5)
            llm_history_reply_logger = logger.bind(user=interaction.user.name)
            llm_history_reply_logger.success("WORDGEN Show History")

    @discord.ui.button(label='Wipe History', emoji="🤯", style=discord.ButtonStyle.grey)
    async def delete_history(self, interaction: discord.Interaction, button: discord.ui.Button):
        """Deletes history"""
        if self.wordobject.user.id == interaction.user.id:
            if await self.wordobject.metatron.is_room_in_queue(self.wordobject.user.id):
                self.wordobject.metatron.generation_queue_concurrency_list[interaction.user.id] += 1
                self.wordobject.action = "wordgenforget"
                await self.wordobject.metatron.generation_queue.put(self.wordobject)
                await interaction.response.send_message("History wiped", ephemeral=True, delete_after=5)
            else:
                await interaction.response.send_message("Queue limit reached, please wait until your current gen or gens finish")
