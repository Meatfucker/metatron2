"""
metatron2 - A discord machine learning bot
"""
import os
import io
import re
import sys
import asyncio
from datetime import datetime
import discord
from discord import app_commands
from loguru import logger
from typing import Optional
from modules.wordgen import Wordgenbuttons, load_llm, llm_generate, clear_history, delete_last_history, insert_history
from modules.speakgen import Speakgenbuttons, load_bark, speak_generate, load_voices
from modules.imagegen import Imagegenbuttons, load_sd, sd_generate, load_models_list, load_ti, load_embeddings_list, \
    load_loras_list, load_sd_lora
from modules.settings import SETTINGS, get_defaults
import torch
import gc
import warnings

warnings.filterwarnings("ignore")
os.environ["TQDM_DISABLE"] = "1"
logger.remove()  # Remove the default configuration

if SETTINGS["enabledebug"][0] == "True":  # this sets up the base logger formatting
    logger.add(sink=io.TextIOWrapper(sys.stdout.buffer, write_through=True),
               format="<light-black>{time:YYYY-MM-DD HH:mm:ss}</light-black> | <level>{level: <8}</level> | <cyan>{name: >16}</cyan>:<light-cyan>{function: <14}</light-cyan> | <light-yellow>{message: ^27}</light-yellow> | <light-red>{extra}</light-red>",
               level="DEBUG", colorize=True)
    logger.add("bot.log", rotation="20 MB",
               format="<light-black>{time:YYYY-MM-DD HH:mm:ss}</light-black> | <level>{level: <8}</level> | <cyan>{name: >8}</cyan>:<light-cyan>{function: <14}</light-cyan> | <light-yellow>{message: ^27}</light-yellow> | <light-red>{extra}</light-red>",
               level="DEBUG", colorize=False)
else:
    logger.add(sink=io.TextIOWrapper(sys.stdout.buffer, write_through=True),
               format="<light-black>{time:YYYY-MM-DD HH:mm:ss}</light-black> | <level>{level: <8}</level> | <light-yellow>{message: ^27}</light-yellow> | <light-red>{extra}</light-red>",
               level="INFO", colorize=True)
    logger.add("bot.log", rotation="20 MB",
               format="<light-black>{time:YYYY-MM-DD HH:mm:ss}</light-black> | <level>{level: <8}</level> | <light-yellow>{message: ^27}</light-yellow> | <light-red>{extra}</light-red>",
               level="INFO", colorize=True)


class MetatronClient(discord.Client):
    """The discord client class for the bot"""

    def __init__(self, *, intents: discord.Intents):
        super().__init__(intents=intents)
        self.slash_command_tree = app_commands.CommandTree(
            self)  # the object that holds the command tree for the slash commands
        self.generation_queue = asyncio.Queue()  # the process queue object
        self.generation_queue_concurrency_list = {}  # A dict to keep track of how many requests each user has queued.
        self.llm_model = None  # this holds the transformers llm model object
        self.llm_tokenizer = None  # this holds the llm tokenizer object
        self.llm_view_last_message = {}  # variable to track word view buttons so there is only one set
        self.llm_chunks_messages = {}  # variable to track the body of the last reply
        self.speak_voices_list = None  # This is a list of available voice files
        self.speak_voice_choices = []  # The choices object for the discord speakgen ui
        self.sd_pipeline = None  # This is the diffusers stable diffusion pipeline object
        self.sd_compel_processor = None  # This is the diffusers compel processor object, which handles prompt weighting
        self.sd_model_list = None  # This is a list of the available model files
        self.sd_model_choices = []  # The choices object for the discord imagegen models ui
        self.sd_loaded_model = None
        self.sd_loaded_embeddings = []  # The list of currently loaded image embeddings
        self.sd_embeddings_list = None  # List of available embeddings
        self.sd_embedding_choices = []  # The choices object for the discord imagegen embeddings ui
        self.sd_loras_list = None  # list of available loras
        self.sd_loras_choices = []  # the choices object for the discord imagegen loras ui

    @logger.catch
    async def setup_hook(self):
        if SETTINGS["enableword"][0] == "True":
            logger.info("Loading LLM")
            self.llm_model, self.llm_tokenizer = await load_llm()  # load llm

        if SETTINGS["enablespeak"][0] == "True":
            logger.info("Loading Bark")
            await load_bark()  # load the sound generation model
            self.speak_voices_list = await load_voices()  # get the list of available voice files to build the discord interface with
            for voice in self.speak_voices_list:
                self.speak_voice_choices.append(app_commands.Choice(name=voice, value=voice))

        if SETTINGS["enableimage"][0] == "True":
            logger.info("Loading SD")
            self.sd_pipeline, self.sd_compel_processor, self.sd_loaded_model = await load_sd()  # load the sd model pipeline and compel prompt processor
            self.sd_model_list = await load_models_list()  # get the list of available models to build the discord interface with
            for model in self.sd_model_list:
                self.sd_model_choices.append(app_commands.Choice(name=model, value=model))
            self.sd_loras_list = await load_loras_list()  # get the list of available loras to build the interface with
            for lora in self.sd_loras_list:
                self.sd_loras_choices.append(app_commands.Choice(name=lora, value=lora))
            self.sd_embeddings_list = await load_embeddings_list()  # get the list of available embeddings to build the discord interface with
            for embedding in self.sd_embeddings_list:
                self.sd_embedding_choices.append(app_commands.Choice(name=embedding, value=embedding))

        self.loop.create_task(client.process_queue())  # start queue
        await self.slash_command_tree.sync()  # sync commands to discord
        logger.info("Logging in...")

    @logger.catch
    async def on_ready(self):
        ready_logger = logger.bind(user=client.user.name, userid=client.user.id)
        ready_logger.info("Login Successful")

    @logger.catch
    async def process_queue(self):
        while True:
            try:
                args = await self.generation_queue.get()
                # logger.debug(f':ARGS: {args}')
                action = args[0]  # first argument passed to queue should always be the action to do
                user_id = args[1]

                if SETTINGS["enableword"][0] == "True":

                    if action == 'wordgenforget':
                        message = args[2]
                        await clear_history(message)
                        llm_clear_history_logger = logger.bind(user=message.author.name, userid=message.author.id)
                        llm_clear_history_logger.success("WORDGEN History Cleared.")

                    elif action == 'wordgengenerate':
                        message, stripped_message, reroll = args[2:5]
                        if reroll:
                            await delete_last_history(message)
                        response = await llm_generate(message, stripped_message, self.llm_model, self.llm_tokenizer)
                        if message.author.id in self.llm_view_last_message:  # check if there are an existing set of llm buttons for the user and if so, delete them
                            try:
                                await self.llm_view_last_message[message.author.id].delete()
                            except discord.NotFound:
                                pass  # Message not found, might have been deleted already
                        if message.author.id in self.llm_chunks_messages:
                            for chunk_message in self.llm_chunks_messages[message.author.id]:
                                if reroll:
                                    try:
                                        await chunk_message.delete()
                                    except discord.NotFound:
                                        pass  # Message not found, might have been deleted already
                            del self.llm_chunks_messages[message.author.id]
                        chunks = [response[i:i + 1500] for i in range(0, len(response), 1500)]
                        if message.author.id not in self.llm_chunks_messages:
                            self.llm_chunks_messages[message.author.id] = []
                        for chunk in chunks:
                            chunk_message = await message.channel.send(chunk)
                            self.llm_chunks_messages[message.author.id].append(chunk_message)
                        new_message = await message.channel.send(
                            view=Wordgenbuttons(self.generation_queue, message.author.id, message, stripped_message, self))  # send the message with the llm buttons
                        self.llm_view_last_message[message.author.id] = new_message  # track the message id of the last set of llm buttons for each user
                        llm_reply_logger = logger.bind(user=message.author.name, prompt=stripped_message)
                        llm_reply_logger.success("WORDGEN Reply")
                        with torch.no_grad():  # clear torch gpu cache, freeing up vram
                            torch.cuda.empty_cache()
                        gc.collect()  # clear python garbage, freeing up ram (and maybe vram?)

                    elif action == 'wordgendeletelast':
                        await delete_last_history(message)
                        llm_delete_last_logger = logger.bind(user=message.author.name, userid=message.author.id)
                        llm_delete_last_logger.success("WORDGEN Reply Deleted.")

                    elif action == 'wordgenimpersonate':
                        prompt, llmprompt, username = args[2:5]
                        await insert_history(user_id, prompt, llmprompt)
                        wordgenreply_logger = logger.bind(user=username, userid=user_id, prompt=prompt, llmprompt=llmprompt)
                        wordgenreply_logger.success("WORDGEN Impersonate.")

                if SETTINGS["enablespeak"][0] == "True":

                    if action == 'speakgengenerate':
                        prompt, channel, voicefile, username = args[2:6]
                        wav_bytes_io = await speak_generate(prompt, voicefile)  # this generates the audio
                        sanitized_prompt = re.sub(r'[^\w\s\-.]', '', prompt)
                        truncatedprompt = sanitized_prompt[:100]  # this avoids file name length limits

                        if SETTINGS["saveinmp3"][0] == "True":
                            await self.save_output(truncatedprompt, wav_bytes_io, "mp3")
                            await channel.send(content=f"Prompt:`{prompt}`",
                                               file=discord.File(wav_bytes_io, filename=f"{truncatedprompt}.mp3"),
                                               view=Speakgenbuttons(self.generation_queue, user_id, prompt, voicefile, self))
                        else:
                            await self.save_output(truncatedprompt, wav_bytes_io, "wav")
                            await channel.send(content=f"Prompt:`{prompt}`",
                                               file=discord.File(wav_bytes_io, filename=f"{truncatedprompt}.wav"),
                                               view=Speakgenbuttons(self.generation_queue, user_id, prompt, voicefile, self))
                        speakgenreply_logger = logger.bind(user=username, prompt=prompt)
                        speakgenreply_logger.success("SPEAKGEN Replied")
                        with torch.no_grad():  # clear gpu memory cache
                            torch.cuda.empty_cache()
                        gc.collect()  # clear python memory

                if SETTINGS["enableimage"][0] == "True":

                    if action == 'imagegenerate':
                        sd_defaults = await get_defaults('global')
                        prompt, channel, sdmodel, batch_size, username, negativeprompt, seed, steps, width, height = args[2:12]

                        if sdmodel is not None:  # if a model has been selected, create and load a fresh pipeline and compel processor
                            self.sd_pipeline, self.sd_compel_processor, self.sd_loaded_model = await load_sd(sdmodel)
                            self.sd_loaded_embeddings = []
                            with torch.no_grad():  # clear gpu memory cache
                                torch.cuda.empty_cache()
                            gc.collect()  # clear python memory
                        else:
                            if self.sd_loaded_model != sd_defaults["imagemodel"][0]:
                                self.sd_pipeline, self.sd_compel_processor, self.sd_loaded_model = await load_sd(sd_defaults["imagemodel"][0])
                                self.sd_loaded_embeddings = []
                            with torch.no_grad():  # clear gpu memory cache
                                torch.cuda.empty_cache()
                            gc.collect()  # clear python memory
                        if batch_size is None:
                            batch_size = int(sd_defaults["imagebatchsize"][0])
                        if steps is None:
                            steps = int(sd_defaults["imagesteps"][0])
                        if width is None:
                            width = int(sd_defaults["imagewidth"][0])
                        if height is None:
                            height = int(sd_defaults["imageheight"][0])
                        if sd_defaults["imageprompt"][0] not in prompt:
                            prompt = f'{sd_defaults["imageprompt"][0]} {prompt}'
                        self.sd_pipeline, prompt_to_gen = await load_sd_lora(self.sd_pipeline, prompt)
                        self.sd_pipeline, loaded_image_embeddings = await load_ti(self.sd_pipeline, prompt_to_gen, self.sd_loaded_embeddings)
                        generated_image = await sd_generate(self.sd_pipeline, self.sd_compel_processor, prompt_to_gen, sdmodel, batch_size, negativeprompt, seed, steps, width,  height)
                        sanitized_prompt = re.sub(r'[^\w\s\-.]', '', prompt)
                        truncatedprompt = sanitized_prompt[:100]  # this avoids file name length limits
                        if SETTINGS["saveinjpg"][0] == "True":
                            await self.save_output(truncatedprompt, generated_image, "jpg")
                        else:
                            await self.save_output(truncatedprompt, generated_image, "png")
                        await channel.send(
                            content=f"Prompt:`{prompt}` Negative:`{negativeprompt}` Model:`{sdmodel}` Batch Size:`{batch_size}` Seed:`{seed}` Steps:`{steps}` Width:`{width}` Height:`{height}` ", file=discord.File(generated_image, filename=f"{truncatedprompt}.png"), view=Imagegenbuttons(self.generation_queue, prompt, channel, sdmodel, batch_size, username, user_id, negativeprompt, seed, steps, width, height, self))
                        imagegenreply_logger = logger.bind(user=username, prompt=prompt, negativeprompt=negativeprompt, model=sdmodel)
                        imagegenreply_logger.success("IMAGEGEN Replied")

                        with torch.no_grad():  # clear gpu memory cache
                            torch.cuda.empty_cache()
                        gc.collect()  # clear python memory
            except Exception as e:
                logger.error(f'EXCEPTION: {e}')
            finally:
                self.generation_queue_concurrency_list[user_id] -= 1

    @logger.catch
    async def is_room_in_queue(self, user_id):
        self.generation_queue_concurrency_list.setdefault(user_id, 0)
        user_queue_depth = int(SETTINGS.get("userqueuedepth", [1])[0])
        if self.generation_queue_concurrency_list[user_id] >= user_queue_depth:
            return False
        else:
            self.generation_queue_concurrency_list[user_id] += 1
            return True

    @logger.catch
    async def save_output(self, prompt, file, filetype):
        current_datetime_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        basepath = f'{SETTINGS["savepath"][0]}/{current_datetime_str}-{prompt}'
        truncatedpath = basepath[:200]
        imagesavepath = f'{truncatedpath}.{filetype}'
        with open(imagesavepath, "wb") as output_file:
            output_file.write(file.getvalue())

    @logger.catch
    async def on_message(self, message):
        if self.user.mentioned_in(message):
            if SETTINGS["enableword"][0] != "True":
                await message.channel.send("LLM generation is currently disabled.")
                return  # check if LLM generation is enabled
            if str(message.author.id) in SETTINGS.get("bannedusers", [""])[0].split(','):
                return  # Exit the function if the author is banned
            stripped_message = re.sub(r'<[^>]+>', '', message.content).lstrip()  # this removes the user tag
            if await self.is_room_in_queue(message.author.id):
                await self.generation_queue.put(
                    ('wordgengenerate', message.author.id, message, stripped_message, False))
            else:
                await message.channel.send("Queue limit has been reached, please wait for your previous gens to finish")


client = MetatronClient(intents=discord.Intents.all())  # client intents


@logger.catch
@client.slash_command_tree.command()
async def impersonate(interaction: discord.Interaction, userprompt: str, llmprompt: str):
    """Slash command that allows for one shot prompting"""
    if SETTINGS["enableword"][0] != "True":
        await interaction.response.send_message("LLM generation is currently disabled.")
        return  # check if LLM generation is enabled
    if str(interaction.user.id) in SETTINGS.get("bannedusers", [""])[0].split(','):
        return  # Exit the function if the author is banned
    if await client.is_room_in_queue(interaction.user.id):
        await client.generation_queue.put(
            ('wordgenimpersonate', interaction.user.id, userprompt, llmprompt, interaction.user.name))
        await interaction.response.send_message(f'History inserted:\n User: {userprompt}\n LLM: {llmprompt}')
    else:
        await interaction.response.send_message(
            "Queue limit reached, please wait until your current gen or gens finish")


@logger.catch
@client.slash_command_tree.command()
@app_commands.choices(voicechoice=client.speak_voice_choices)
@app_commands.rename(userprompt='prompt', voicechoice='voice')
async def speakgen(interaction: discord.Interaction, userprompt: str,
                   voicechoice: Optional[app_commands.Choice[str]] = None):
    """Slash command that generates speech"""
    if SETTINGS["enablespeak"][0] != "True":
        await interaction.response.send_message("Sound generation is currently disabled.")
        return  # check if sound generation is enabled
    if str(interaction.user.id) in SETTINGS.get("bannedusers", [""])[0].split(','):
        return  # Exit the function if the author is banned
    if voicechoice is None:
        voiceselection = None
    else:
        voiceselection = voicechoice.name
    if await client.is_room_in_queue(interaction.user.id):
        await interaction.response.send_message("Generating Sound...", ephemeral=True, delete_after=5)
        await client.generation_queue.put(('speakgengenerate', interaction.user.id, userprompt, interaction.channel, voiceselection, interaction.user.name))
    else:
        await interaction.response.send_message("Queue limit reached, please wait until your current gen or gens finish")


@logger.catch
@client.slash_command_tree.command()
@app_commands.choices(modelchoice=client.sd_model_choices)
@app_commands.choices(embeddingchoice=client.sd_embedding_choices)
@app_commands.choices(lorachoice=client.sd_loras_choices)
@app_commands.rename(userprompt='prompt', modelchoice='model', embeddingchoice='embedding', lorachoice='lora')
async def imagegen(interaction: discord.Interaction, userprompt: str, negativeprompt: Optional[str], modelchoice: Optional[app_commands.Choice[str]] = None, lorachoice: Optional[app_commands.Choice[str]] = None, embeddingchoice: Optional[app_commands.Choice[str]] = None, batch_size: Optional[int] = None, seed: Optional[int] = None, steps: Optional[int] = None, width: Optional[int] = None, height: Optional[int] = None):
    """Slash command that generates images"""
    if SETTINGS["enableimage"][0] != "True":
        await interaction.response.send_message("Image generation is currently disabled.")
        return  # check if image generation is enabled
    if str(interaction.user.id) in SETTINGS.get("bannedusers", [""])[0].split(','):
        return  # Exit the function if the author is banned
    if modelchoice is None:
        modelselection = None
    else:
        modelselection = modelchoice.name
    if lorachoice is not None:
        userprompt = f"{userprompt}<lora:{lorachoice.name}:1>"
    if embeddingchoice is not None:
        userprompt = f"{userprompt}{embeddingchoice.name}"
    if await client.is_room_in_queue(interaction.user.id):
        await interaction.response.send_message("Generating Image...", ephemeral=True, delete_after=5)
        await client.generation_queue.put(('imagegenerate', interaction.user.id, userprompt, interaction.channel, modelselection, batch_size, interaction.user.name, negativeprompt, seed, steps, width, height))
    else:
        await interaction.response.send_message(
            "Queue limit reached, please wait until your current gen or gens finish")


async def quit_exit():
    # Perform cleanup tasks here if needed before exiting
    # Close resources, finish ongoing tasks, etc.
    logger.info("Shutting down.")
    await client.close()  # Close the Discord bot connection gracefully
    sys.exit(0)  # Exit the program


def run_program():
    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(client.start(SETTINGS["token"][0]))  # Start the bot
    except KeyboardInterrupt:
        loop.run_until_complete(
            quit_exit())  # If KeyboardInterrupt occurs during setup or start, perform cleanup and exit
    finally:
        loop.close()


if __name__ == "__main__":
    run_program()
