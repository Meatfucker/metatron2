"""
metatron2 - A discord machine learning bot
"""
import io
import sys
import asyncio
import discord
import re
import gc
import torch
from discord import app_commands
from loguru import logger
from typing import Optional
import numpy as np
from modules.speakgen import VoiceQueueObject, load_bark, load_voices
from modules.wordgen import WordQueueObject, load_llm
from modules.imagegen import ImageQueueObject, load_models_list, load_embeddings_list, load_loras_list
from modules.imagegen_xl import ImageXLQueueObject, load_sdxl_models_list, load_sdxl_loras_list, load_sdxl_refiners_list
from modules.voiceclone import CloneQueueObject
from modules.api import ApiImageQueueObject, ApiWordQueueObject, api_load_sd_models, api_load_sd_loras
from modules.settings import SETTINGS
import warnings

warnings.filterwarnings("ignore")
logger.remove()  # Remove the default configuration

if SETTINGS["enabledebug"][0] == "True":  # this sets up the base logger formatting
    logger.add(sink=io.TextIOWrapper(sys.stdout.buffer, write_through=True), format="<light-black>{time:YYYY-MM-DD HH:mm:ss}</light-black> | <level>{level: <8}</level> | <cyan>{name: >16}</cyan>:<light-cyan>{function: <14}</light-cyan> | <light-yellow>{message: ^27}</light-yellow> | <light-red>{extra}</light-red>", level="DEBUG", colorize=True)
    logger.add("bot.log", rotation="20 MB", format="<light-black>{time:YYYY-MM-DD HH:mm:ss}</light-black> | <level>{level: <8}</level> | <cyan>{name: >8}</cyan>:<light-cyan>{function: <14}</light-cyan> | <light-yellow>{message: ^27}</light-yellow> | <light-red>{extra}</light-red>", level="DEBUG", colorize=False)
else:
    logger.add(sink=io.TextIOWrapper(sys.stdout.buffer, write_through=True), format="<light-black>{time:YYYY-MM-DD HH:mm:ss}</light-black> | <level>{level: <8}</level> | <light-yellow>{message: ^27}</light-yellow> | <light-red>{extra}</light-red>", level="INFO", colorize=True)
    logger.add("bot.log", rotation="20 MB", format="<light-black>{time:YYYY-MM-DD HH:mm:ss}</light-black> | <level>{level: <8}</level> | <light-yellow>{message: ^27}</light-yellow> | <light-red>{extra}</light-red>", level="INFO", colorize=True)


class MetatronClient(discord.Client):
    """The discord client class for the bot"""

    def __init__(self, *, intents: discord.Intents):
        super().__init__(intents=intents)
        self.slash_command_tree = app_commands.CommandTree(self)  # the object that holds the command tree for the slash commands
        self.generation_queue = asyncio.Queue()  # the process queue object
        self.generation_queue_concurrency_list = {}  # A dict to keep track of how many requests each user has queued.

        self.speak_voice_choices = []  # The choices object for the discord speakgen ui

        self.llm_model = None
        self.llm_user_history = {}
        self.llm_view_last_message = {}  # variable to track word view buttons so there is only one set
        self.llm_chunks_messages = {}  # variable to track the body of the last reply

        self.sd_pipeline = None  # This is the diffusers stable diffusion pipeline object
        self.sd_compel_processor = None  # This is the diffusers compel processor object, which handles prompt weighting
        self.sd_model_choices = []  # The choices object for the discord imagegen models ui
        self.sd_loaded_model = None
        self.sd_loaded_embeddings = []  # The list of currently loaded image embeddings
        self.sd_embedding_choices = []  # The choices object for the discord imagegen embeddings ui
        self.sd_loras_choices = []  # the choices object for the discord imagegen loras ui

        self.sd_xl_pipeline = None
        self.sd_xl_compel_processor = None
        self.sd_xl_model_choices = []
        self.sd_xl_loras_choices = []
        self.sd_xl_loaded_model = None
        self.sd_xl_loaded_refiner = None

    async def setup_hook(self):
        """This loads the various models before logging in to discord"""

        if SETTINGS["enableword"][0] == "True":
            if SETTINGS["enablewordapi"][0] != "True":
                logger.info("Loading LLM")
                self.llm_model = await load_llm()  # load llm

        if SETTINGS["enablespeak"][0] == "True":
            logger.info("Loading Bark")
            await load_bark()  # load the sound generation model
            speak_voices_list = await load_voices()  # get the list of available voice files to build the discord interface with
            for voice in speak_voices_list:
                self.speak_voice_choices.append(app_commands.Choice(name=voice, value=voice))

        if SETTINGS["enablesd"][0] == "True":
            if SETTINGS["enableimageapi"][0] == "True":
                sd_model_list = await api_load_sd_models()  # get the list of available models to build the discord interface with
                for model in sd_model_list:
                    self.sd_model_choices.append(app_commands.Choice(name=model, value=model))
                sd_loras_list = await api_load_sd_loras()  # get the list of available loras to build the interface with
                for lora in sd_loras_list:
                    self.sd_loras_choices.append(app_commands.Choice(name=lora, value=lora))

            else:
                sd_model_list = await load_models_list()  # get the list of available models to build the discord interface with
                for model in sd_model_list:
                    self.sd_model_choices.append(app_commands.Choice(name=model, value=model))
                sd_loras_list = await load_loras_list()  # get the list of available loras to build the interface with
                for lora in sd_loras_list:
                    self.sd_loras_choices.append(app_commands.Choice(name=lora, value=lora))
                sd_embeddings_list = await load_embeddings_list()  # get the list of available embeddings to build the discord interface with
                for embedding in sd_embeddings_list:
                    self.sd_embedding_choices.append(app_commands.Choice(name=embedding, value=embedding))

        if SETTINGS["enablesdxl"][0] == "True":
            sdxl_model_list = await load_sdxl_models_list()  # get the list of available models to build the discord interface with
            for model in sdxl_model_list:
                self.sd_xl_model_choices.append(app_commands.Choice(name=model, value=model))
            sd_xl_loras_list = await load_sdxl_loras_list()  # get the list of available loras to build the interface with
            for lora in sd_xl_loras_list:
                self.sd_xl_loras_choices.append(app_commands.Choice(name=lora, value=lora))

        self.loop.create_task(client.process_queue())  # start queue
        logger.info("Logging in...")

    async def on_ready(self):
        """Just prints the bots name to discord"""
        await self.slash_command_tree.sync()  # sync commands to discord
        ready_logger = logger.bind(user=client.user.name, userid=client.user.id)
        ready_logger.info("Login Successful")

    @logger.catch()
    async def on_message(self, message):
        """This captures people talking to the bot in chat and responds."""
        if self.user.mentioned_in(message):
            if not await self.is_enabled_not_banned("enableword", message.author):
                return
            prompt = re.sub(r'<[^>]+>', '', message.content).lstrip()  # this removes the user tag
            image_urls = None
            if message.attachments:
                image_urls = [attachment.url for attachment in message.attachments]
            else:
                image_url_pattern = r'\bhttps?://\S+\.(?:png|jpg|jpeg|gif)\S*\b'  # Updated regex pattern for image URLs
                image_urls = re.findall(image_url_pattern, prompt)
            if await self.is_room_in_queue(message.author.id):
                self.generation_queue_concurrency_list[message.author.id] += 1
                if SETTINGS["enablewordapi"][0] == "True":
                    wordgen_request = ApiWordQueueObject("wordgen", self, message.author, message.channel, prompt)
                else:
                    wordgen_request = WordQueueObject("wordgen", self, message.author, message.channel, prompt, image_urls)
                await self.generation_queue.put(wordgen_request)
            else:
                await message.channel.send("Queue limit has been reached, please wait for your previous gens to finish")

    async def process_queue(self):
        """This is the primary queue for the bot. Anything that requires state be maintained goes through here"""
        while True:
            queue_request = await self.generation_queue.get()
            try:

                if SETTINGS["enablespeak"][0] == "True":

                    if queue_request.action == "speakgen":
                        await queue_request.generate()
                        if SETTINGS["saveoutputs"][0] == "True":
                            await queue_request.save()
                        await queue_request.respond()

                    if queue_request.action == "voiceclone":
                        if await queue_request.check_audio_format():
                            if await queue_request.check_audio_duration():
                                await queue_request.clone_voice()
                                await queue_request.respond()

                if SETTINGS["enableword"][0] == "True":

                    if queue_request.action == "wordgen":
                        await queue_request.generate()
                        await queue_request.respond()

                    if queue_request.action == "wordgendeletelast":
                        await queue_request.delete_last_history_pair()

                    if queue_request.action == "wordgenforget":
                        await queue_request.clear_history()

                    if queue_request.action == "wordgensummary":
                        await queue_request.summary()

                    if queue_request.action == "wordgeninject":
                        await queue_request.insert_history()

                if SETTINGS["enablesd"][0] == "True":

                    if queue_request.action == "imagegen":
                        await queue_request.generate()
                        with torch.no_grad():  # clear gpu memory cache
                            torch.cuda.empty_cache()
                        gc.collect()  # clear python memory
                        if SETTINGS["saveoutputs"][0] == "True":
                            await queue_request.save()
                        await queue_request.respond()

                if SETTINGS["enablesdxl"][0] == "True":

                    if queue_request.action == "xlimagegen":
                        await queue_request.generate()
                        with torch.no_grad():  # clear gpu memory cache
                            torch.cuda.empty_cache()
                        gc.collect()  # clear python memory
                        if SETTINGS["saveoutputs"][0] == "True":
                            await queue_request.save()
                        await queue_request.respond()

            except Exception as e:
                logger.error(f'EXCEPTION: {e}')
            finally:
                self.generation_queue_concurrency_list[queue_request.user.id] -= 1  # Remove one from the users queue limit
                with torch.no_grad():  # clear gpu memory cache
                    torch.cuda.empty_cache()
                gc.collect()  # clear python memory

    async def is_room_in_queue(self, user_id):
        """This checks the users current number of pending gens against the max, and if there is room, returns true, otherwise, false"""
        self.generation_queue_concurrency_list.setdefault(user_id, 0)
        user_queue_depth = int(SETTINGS.get("userqueuedepth", [1])[0])
        if self.generation_queue_concurrency_list[user_id] >= user_queue_depth:
            return False
        else:
            return True

    async def is_enabled_not_banned(self, module, user):
        """This only returns true if the module is both enabled and the user is not banned"""
        if SETTINGS[module][0] != "True":
            return False  # check if LLM generation is enabled
        elif str(user.id) in SETTINGS.get("bannedusers", [""])[0].split(','):
            return False  # Exit the function if the author is banned
        else:
            return True


client = MetatronClient(intents=discord.Intents.all())  # client intents


@client.slash_command_tree.command(description="SDXL image generation")
@app_commands.describe(prompt="The prompt for text encoder 1", prompt_2="The prompt for text encoder 2, if blank prompt is used.", negative_prompt="The negative prompt for text encoder 1",
                       negative_prompt_2="The negative prompt for text encoder 2", model_choice="The model to use for generation", lora_choice="The lora to use in generation",
                       batch_size="The number of images to generate", seed="The generation seed, if blank a random one is used", steps="The number of inference steps", width="Image width",
                       height="Image height", use_defaults="Use channel defaults?")
@app_commands.choices(model_choice=client.sd_xl_model_choices)
@app_commands.choices(lora_choice=client.sd_xl_loras_choices)
async def xl_imagegen(interaction: discord.Interaction, prompt: str, prompt_2: Optional[str], negative_prompt: Optional[str], negative_prompt_2: Optional[str],
                      model_choice: Optional[app_commands.Choice[str]] = None, lora_choice: Optional[app_commands.Choice[str]] = None, batch_size: Optional[int] = None,
                      seed: Optional[int] = None, steps: Optional[int] = None, width: Optional[int] = None, height: Optional[int] = None, use_defaults: bool = True):
    """This is the slash command for imagegen."""
    if not await client.is_enabled_not_banned("enablesdxl", interaction.user):
        await interaction.response.send_message("SD disabled or user banned", ephemeral=True, delete_after=5)
        return
    if model_choice is None:
        model_selection = None
    else:
        model_selection = model_choice.name
    if lora_choice is not None:
        prompt = f"{prompt}<lora:{lora_choice.name}:1>"

    xlimagegen_request = ImageXLQueueObject("xlimagegen", client, interaction.user, interaction.channel, prompt, prompt_2, negative_prompt, negative_prompt_2, model_selection, batch_size, seed, steps, width, height, use_defaults)
    if await client.is_room_in_queue(interaction.user.id):
        await interaction.response.send_message("Generating Image...", ephemeral=True, delete_after=5)
        client.generation_queue_concurrency_list[interaction.user.id] += 1
        await client.generation_queue.put(xlimagegen_request)
    else:
        await interaction.response.send_message(
            "Queue limit reached, please wait until your current gen or gens finish")


@client.slash_command_tree.command(description="Stable Diffusion image generation")
@app_commands.describe(prompt="Prompt for generation", negative_prompt="Negative prompt for generation", model_choice="The model to use for generation", lora_choice="The lora to use for generation",
                       embedding_choice="The embedding to use for generation", batch_size="Number of images to generate", seed="The seed to use for generation, if blank, a random one is used",
                       steps="The steps to use for generation", width="Image width", height="Image height", use_defaults="Use channel defaults?")
@app_commands.choices(model_choice=client.sd_model_choices)
@app_commands.choices(embedding_choice=client.sd_embedding_choices)
@app_commands.choices(lora_choice=client.sd_loras_choices)
async def imagegen(interaction: discord.Interaction, prompt: str, negative_prompt: Optional[str], model_choice: Optional[app_commands.Choice[str]] = None,
                   lora_choice: Optional[app_commands.Choice[str]] = None, embedding_choice: Optional[app_commands.Choice[str]] = None, batch_size: Optional[int] = None,
                   seed: Optional[int] = None, steps: Optional[int] = None, width: Optional[int] = None, height: Optional[int] = None, use_defaults: bool = True):
    """This is the slash command for imagegen."""
    if not await client.is_enabled_not_banned("enablesd", interaction.user):
        await interaction.response.send_message("SD disabled or user banned", ephemeral=True, delete_after=5)
        return
    if model_choice is None:
        model_selection = None
    else:
        model_selection = model_choice.name
    if lora_choice is not None:
        prompt = f"{prompt}<lora:{lora_choice.name}:1>"
    if embedding_choice is not None:
        prompt = f"{prompt}{embedding_choice.name}"
    if SETTINGS["enableimageapi"][0] == "True":
        imagegen_request = ApiImageQueueObject("imagegen", client, interaction.user, interaction.channel, prompt, negative_prompt, model_selection, batch_size, seed, steps, width, height, use_defaults)
    else:
        imagegen_request = ImageQueueObject("imagegen", client, interaction.user, interaction.channel, prompt, negative_prompt, model_selection, batch_size, seed, steps, width, height, use_defaults)
    if await client.is_room_in_queue(interaction.user.id):
        await interaction.response.send_message("Generating Image...", ephemeral=True, delete_after=5)
        client.generation_queue_concurrency_list[interaction.user.id] += 1
        await client.generation_queue.put(imagegen_request)
    else:
        await interaction.response.send_message(
            "Queue limit reached, please wait until your current gen or gens finish")


@client.slash_command_tree.command(description="This is used to insert message/reply pairs into your LLM user history")
@app_commands.describe(user_prompt="Your sentence or statement", llm_prompt="The LLMs reply")
async def impersonate(interaction: discord.Interaction, user_prompt: str, llm_prompt: str):
    """This is the slash command for impersonate"""
    if not await client.is_enabled_not_banned("enableword", interaction.user):
        await interaction.response.send_message("LLM disabled or user banned", ephemeral=True, delete_after=5)
        return
    wordgen_request = WordQueueObject("wordgeninject", client, interaction.user, interaction.channel, user_prompt, None, llm_prompt)
    if await client.is_room_in_queue(interaction.user.id):
        client.generation_queue_concurrency_list[interaction.user.id] += 1
        await client.generation_queue.put(wordgen_request)
        await interaction.response.send_message(f'History inserted:\n User: {user_prompt}\n LLM: {llm_prompt}')
    else:
        await interaction.response.send_message(
            "Queue limit reached, please wait until your current gen or gens finish")


@client.slash_command_tree.command(description="LLM summary of the chatroom.")
async def summarize(interaction: discord.Interaction):
    """This is the slash command for wordgen"""
    if not await client.is_enabled_not_banned("enableword", interaction.user):
        await interaction.response.send_message("LLM disabled or user banned", ephemeral=True, delete_after=5)
        return
    if SETTINGS["enablewordapi"][0] == "True":
        wordgen_request = ApiWordQueueObject("wordgensummary", client, interaction.user, interaction.channel)
    else:
        wordgen_request = WordQueueObject("wordgensummary", client, interaction.user, interaction.channel)
    if await client.is_room_in_queue(interaction.user.id):
        await interaction.response.send_message("Summarizing...")
        client.generation_queue_concurrency_list[interaction.user.id] += 1
        await client.generation_queue.put(wordgen_request)
    else:
        await interaction.response.send_message("Queue limit reached, please wait until your current gen or gens finish", ephemeral=True, delete_after=5)


@client.slash_command_tree.command(description="Bark audio generation")
@app_commands.describe(prompt="The sentence or sounds to generate. use [] around words for noises or sound effects, use ♪ for music",
                       voice_file="The voice to use for generation, if blank the baseline voice is used",
                       user_voice_file="A .npz file generated with /voiceclone")
@app_commands.choices(voice_file=client.speak_voice_choices)
@app_commands.rename(prompt='prompt', voice_file='voice')
async def speakgen(interaction: discord.Interaction, prompt: str, voice_file: Optional[app_commands.Choice[str]] = None, user_voice_file: Optional[discord.Attachment] = None):
    """This is the slash command for speakgen."""
    if not await client.is_enabled_not_banned("enablespeak", interaction.user):
        await interaction.response.send_message("Bark disabled or user banned", ephemeral=True, delete_after=5)
        return
    if user_voice_file is not None:
        input_data = await user_voice_file.read()
        input_file = io.BytesIO(input_data)
        input_file.seek(0)
        try:
            # Attempt to load the file as an npz file
            npz_file = np.load(input_file)
            # If successfully loaded without errors, check its attributes
            if isinstance(npz_file, np.lib.npyio.NpzFile):
                speakgen_request = VoiceQueueObject("speakgen", client, interaction.user, interaction.channel, prompt, voice_file, user_voice_file)
            else:
                await interaction.response.send_message("Supplied speaker file is not a voice.")
                return

        except Exception as e:
            await interaction.response.send_message("Supplied speaker file is not a voice.")
            return
    else:
        speakgen_request = VoiceQueueObject("speakgen", client, interaction.user, interaction.channel, prompt, voice_file)
    if await client.is_room_in_queue(interaction.user.id):
        await interaction.response.send_message("Generating Sound...", ephemeral=True, delete_after=5)
        client.generation_queue_concurrency_list[interaction.user.id] += 1
        await client.generation_queue.put(speakgen_request)
    else:
        await interaction.response.send_message("Queue limit reached, please wait until your current gen or gens finish", ephemeral=True, delete_after=5)


@client.slash_command_tree.command(description="Voice cloning")
@app_commands.describe(user_audio_file="The audio file to clone, must be wav or mp3 and no longer than 30 seconds")
async def voiceclone(interaction: discord.Interaction, user_audio_file: discord.Attachment):
    if not await client.is_enabled_not_banned("enablespeak", interaction.user):
        await interaction.response.send_message("Bark disabled or user banned", ephemeral=True, delete_after=5)
        return

    input_data = await user_audio_file.read()
    input_file = io.BytesIO(input_data)
    input_file.seek(0)
    voiceclone_request = CloneQueueObject("voiceclone", client, interaction.user, interaction.channel, input_file, user_audio_file.filename)
    if await client.is_room_in_queue(interaction.user.id):
        await interaction.response.send_message("Cloning voice...", ephemeral=True, delete_after=5)
        client.generation_queue_concurrency_list[interaction.user.id] += 1
        await client.generation_queue.put(voiceclone_request)
    else:
        await interaction.response.send_message("Queue limit reached, please wait until your current gen or gens finish", ephemeral=True, delete_after=5)


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
        loop.run_until_complete(quit_exit())  # If KeyboardInterrupt occurs during setup or start, perform cleanup and exit
    finally:
        loop.close()


if __name__ == "__main__":
    run_program()
