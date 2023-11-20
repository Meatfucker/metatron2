"""
metatron2 - A discord machine learning bot
"""
import io
import re
import sys
import asyncio
from datetime import datetime
import discord
from discord import app_commands
from loguru import logger
from typing import Optional
from modules.wordgen import Wordgenbuttons, load_llm, llm_generate, clear_history, delete_last_history, insert_history, llm_summary
from modules.speakgen import Speakgenbuttons, load_bark, speak_generate, load_voices
from modules.imagegen import Imagegenbuttons, load_sd, sd_generate, load_models_list, load_ti, load_embeddings_list, \
    load_loras_list, load_sd_lora
from modules.settings import SETTINGS, get_defaults
import torch
import gc
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

    async def setup_hook(self):
        """This loads the various models before logging in to discord"""
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

    async def on_ready(self):
        """Just prints the bots name to discord"""
        ready_logger = logger.bind(user=client.user.name, userid=client.user.id)
        ready_logger.info("Login Successful")

    async def process_queue(self):
        """This is the primary queue for the bot. Anything that requires state be maintained goes through here"""
        while True:
            args = await self.generation_queue.get()
            action = args[0]  # first argument passed to queue should always be the action to do
            user_id = args[1]  # second should always be the user id.
            try:
                if SETTINGS["enableword"][0] == "True":

                    if action == 'wordgenforget':
                        user = args[2]
                        await clear_history(user)
                        llm_clear_history_logger = logger.bind(user=user.name, userid=user.id)
                        llm_clear_history_logger.success("WORDGEN History Cleared.")

                    elif action == 'wordgengenerate':
                        channel, user, prompt, negative_prompt, reroll = args[2:7]
                        await self.queue_wordgen(channel, user, prompt, negative_prompt, reroll)
                        with torch.no_grad():  # clear torch gpu cache, freeing up vram
                            torch.cuda.empty_cache()
                        gc.collect()  # clear python garbage, freeing up ram (and maybe vram?)

                    elif action == 'wordgensummary':
                        channel, user, prompt = args[2:5]
                        await self.queue_summary(channel, user, prompt)
                        with torch.no_grad():  # clear torch gpu cache, freeing up vram
                            torch.cuda.empty_cache()
                        gc.collect()  # clear python garbage, freeing up ram (and maybe vram?)

                    elif action == 'wordgendeletelast':
                        user = args[2]
                        await delete_last_history(user)
                        llm_delete_last_logger = logger.bind(user=user.name, userid=user.id)
                        llm_delete_last_logger.success("WORDGEN Reply Deleted.")

                    elif action == 'wordgenimpersonate':
                        prompt, llmprompt, username = args[2:5]
                        await insert_history(user_id, prompt, llmprompt)
                        wordgenreply_logger = logger.bind(user=username, userid=user_id, prompt=prompt, llmprompt=llmprompt)
                        wordgenreply_logger.success("WORDGEN Impersonate.")

                if SETTINGS["enablespeak"][0] == "True":

                    if action == 'speakgengenerate':
                        prompt, channel, voicefile, user = args[2:6]
                        await self.queue_speak(prompt, channel, voicefile, user)
                        with torch.no_grad():  # clear gpu memory cache
                            torch.cuda.empty_cache()
                        gc.collect()  # clear python memory

                if SETTINGS["enableimage"][0] == "True":

                    if action == 'imagegenerate':
                        prompt, channel, sdmodel, batch_size, username, negativeprompt, seed, steps, width, height, use_defaults = args[2:13]
                        await self.queue_image(prompt, channel, sdmodel, batch_size, username, negativeprompt, seed, steps, width, height, user_id, use_defaults)
                        with torch.no_grad():  # clear gpu memory cache
                            torch.cuda.empty_cache()
                        gc.collect()  # clear python memory

            except Exception as e:
                logger.error(f'EXCEPTION: {e}')
                logger.error(f'ARGS: {args}')
            finally:
                self.generation_queue_concurrency_list[user_id] -= 1  # Remove one from the users queue limit

    async def queue_image(self, prompt, channel, sdmodel, batch_size, username, negativeprompt, seed, steps, width, height, user_id, use_defaults):
        """The queue function that calls the image generation"""
        if use_defaults is True:
            channel_defaults = await get_defaults(channel.id)
            if channel_defaults is not None:
                sd_defaults = channel_defaults
        else:
            sd_defaults = await get_defaults('global')

        if sdmodel is not None:  # if a model has been selected, create and load a fresh pipeline and compel processor
            if self.sd_loaded_model != sdmodel:  #Only load the model if we dont already have it loaded
                self.sd_pipeline, self.sd_compel_processor, self.sd_loaded_model = await load_sd(sdmodel)
                self.sd_loaded_embeddings = []  # Since we loaded a new model, clear the loaded embeddings list
                with torch.no_grad():  # clear gpu memory cache
                    torch.cuda.empty_cache()
                gc.collect()  # clear python memory
        else:
            if self.sd_loaded_model != sd_defaults["imagemodel"][0]:  # If the current model isnt the default model, load it.
                self.sd_pipeline, self.sd_compel_processor, self.sd_loaded_model = await load_sd(sd_defaults["imagemodel"][0])
                self.sd_loaded_embeddings = []  # Since we loaded a new model, clear the loaded embeddings list
            with torch.no_grad():  # clear gpu memory cache
                torch.cuda.empty_cache()
            gc.collect()  # clear python memory

        if batch_size is None:  # Enforce various defaults
            batch_size = int(sd_defaults["imagebatchsize"][0])
        if steps is None:
            steps = int(sd_defaults["imagesteps"][0])
        if width is None:
            width = int(sd_defaults["imagewidth"][0])
        if height is None:
            height = int(sd_defaults["imageheight"][0])

        if prompt is not None:
            if sd_defaults["imageprompt"][0] not in prompt:  # Combine the defaults with the users prompt and negative prompt.
                prompt = f'{sd_defaults["imageprompt"][0]} {prompt}'
        if negativeprompt is not None:
            if sd_defaults["imagenegprompt"][0] not in negativeprompt:
                negativeprompt = f'{sd_defaults["imagenegprompt"][0]} {negativeprompt}'
        else:
            negativeprompt = sd_defaults["imagenegprompt"][0]

        self.sd_pipeline, prompt_to_gen = await load_sd_lora(self.sd_pipeline, prompt)  # Check the prompt for loras and load them if needed.
        self.sd_pipeline, loaded_image_embeddings = await load_ti(self.sd_pipeline, prompt_to_gen, self.sd_loaded_embeddings)  # Check the prompt for TIs and load them if needed.
        generated_image = await sd_generate(self.sd_pipeline, self.sd_compel_processor, prompt_to_gen, sdmodel, batch_size, negativeprompt, seed, steps, width, height)  # Generate the image

        sanitized_prompt = re.sub(r'[^\w\s\-.]', '', prompt)  # Make the prompt safe to be a filename.
        truncatedprompt = sanitized_prompt[:100]  # this avoids file name length limits
        if SETTINGS["saveinjpg"][0] == "True":  # Save and upload in jpg if enabled, otherwise PNG
            await self.save_output(truncatedprompt, generated_image, "jpg")
            await channel.send(content=f"Prompt:`{prompt}` Negative:`{negativeprompt}` Model:`{sdmodel}` Batch Size:`{batch_size}` Seed:`{seed}` Steps:`{steps}` Width:`{width}` Height:`{height}` ", file=discord.File(generated_image, filename=f"{truncatedprompt}.jpg"), view=Imagegenbuttons(self.generation_queue, prompt, channel, sdmodel, batch_size, username, user_id, negativeprompt, steps, width, height, self, use_defaults))
        else:
            await self.save_output(truncatedprompt, generated_image, "png")
            await channel.send(content=f"Prompt:`{prompt}` Negative:`{negativeprompt}` Model:`{sdmodel}` Batch Size:`{batch_size}` Seed:`{seed}` Steps:`{steps}` Width:`{width}` Height:`{height}` ", file=discord.File(generated_image, filename=f"{truncatedprompt}.png"), view=Imagegenbuttons(self.generation_queue, prompt, channel, sdmodel, batch_size, username, user_id, negativeprompt, steps, width, height, self, use_defaults))
        imagegenreply_logger = logger.bind(user=username, prompt=prompt, negativeprompt=negativeprompt, model=sdmodel)
        imagegenreply_logger.success("IMAGEGEN Replied")
        return

    async def queue_speak(self, prompt, channel, voicefile, user):
        """This is the queue function that generates audio"""
        wav_bytes_io = await speak_generate(prompt, voicefile)  # this generates the audio

        sanitized_prompt = re.sub(r'[^\w\s\-.]', '', prompt)  # This makes the prompt safe to be a filename
        truncatedprompt = sanitized_prompt[:100]  # this avoids file name length limits
        if SETTINGS["saveinmp3"][0] == "True":  # If enabled, save and upload in mp3, otherwise wav.
            await self.save_output(truncatedprompt, wav_bytes_io, "mp3")
            await channel.send(content=f"Prompt:`{prompt}`", file=discord.File(wav_bytes_io, filename=f"{truncatedprompt}.mp3"), view=Speakgenbuttons(self.generation_queue, user.id, prompt, voicefile, self))
        else:
            await self.save_output(truncatedprompt, wav_bytes_io, "wav")
            await channel.send(content=f"Prompt:`{prompt}`", file=discord.File(wav_bytes_io, filename=f"{truncatedprompt}.wav"), view=Speakgenbuttons(self.generation_queue, user.id, prompt, voicefile, self))
        speakgenreply_logger = logger.bind(user=user.name, prompt=prompt)
        speakgenreply_logger.success("SPEAKGEN Replied")
        return

    async def queue_wordgen(self, channel, user, prompt, negative_prompt, reroll):
        """This is the queue function that generates chat"""
        if reroll:  # If its a reroll, delete the last history pair.
            await delete_last_history(user)

        response = await llm_generate(user, prompt, negative_prompt, self.llm_model, self.llm_tokenizer) # Generate the text

        if user.id in self.llm_view_last_message:  # check if there are an existing set of llm buttons for the user and if so, delete them
            try:
                await self.llm_view_last_message[user.id].delete()
            except discord.NotFound:
                pass

        if user.id in self.llm_chunks_messages:  # If its a reroll, delete the old messages
            for chunk_message in self.llm_chunks_messages[user.id]:
                if reroll:
                    try:
                        await chunk_message.delete()
                    except discord.NotFound:
                        pass  # Message not found, might have been deleted already
            del self.llm_chunks_messages[user.id]

        chunks = [response[i:i + 1500] for i in range(0, len(response), 1500)]  # Send and track the previously sent messages in case we have to delete them for reroll.
        if user.id not in self.llm_chunks_messages:
            self.llm_chunks_messages[user.id] = []
        for chunk in chunks:
            chunk_message = await channel.send(chunk)
            self.llm_chunks_messages[user.id].append(chunk_message)
        new_message = await channel.send(view=Wordgenbuttons(self.generation_queue, user.id, prompt, negative_prompt, self))  # send the message with the llm buttons
        self.llm_view_last_message[user.id] = new_message  # track the message id of the last set of llm buttons for each user
        llm_reply_logger = logger.bind(user=user.name, prompt=prompt, negative=negative_prompt)
        llm_reply_logger.success("WORDGEN Reply")
        return

    async def queue_summary(self, channel, user, prompt):
        """This is the queue function that summarizes chat."""
        response = await llm_summary(user, prompt, self.llm_model, self.llm_tokenizer)  # Generate the summary.

        chunks = [response[i:i + 1500] for i in range(0, len(response), 1500)]  # Post the message
        for chunk in chunks:
            await channel.send(chunk)
        llm_summary_logger = logger.bind(user=user.name)
        llm_summary_logger.success("SUMMARY Reply")
        return

    async def is_room_in_queue(self, user_id):
        """This checks the users current number of pending gens against the max, and if there is room, returns true, otherwise, false"""
        self.generation_queue_concurrency_list.setdefault(user_id, 0)
        user_queue_depth = int(SETTINGS.get("userqueuedepth", [1])[0])
        if self.generation_queue_concurrency_list[user_id] >= user_queue_depth:
            return False
        else:
            return True

    async def save_output(self, prompt, file, filetype):
        """This saves whatever file-like object its given with the extension its given, based on its prompt"""
        if SETTINGS["saveoutputs"][0] == "True":
            current_datetime_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            basepath = f'{SETTINGS["savepath"][0]}/{current_datetime_str}-{prompt}'
            truncatedpath = basepath[:200]
            imagesavepath = f'{truncatedpath}.{filetype}'
            with open(imagesavepath, "wb") as output_file:
                output_file.write(file.getvalue())
        return

    async def is_enabled_not_banned(self, module, user):
        """This only returns true if the module is both enabled and the user is not banned"""
        if SETTINGS[module][0] != "True":
            return False  # check if LLM generation is enabled
        elif str(user.id) in SETTINGS.get("bannedusers", [""])[0].split(','):
            return False  # Exit the function if the author is banned
        else:
            return True

    async def on_message(self, message):
        """This captures people talking to the bot in chat and responds."""
        if self.user.mentioned_in(message):
            if not await self.is_enabled_not_banned("enableword", message.author):
                return
            prompt = re.sub(r'<[^>]+>', '', message.content).lstrip()  # this removes the user tag
            if await self.is_room_in_queue(message.author.id):
                self.generation_queue_concurrency_list[message.author.id] += 1
                await self.generation_queue.put(('wordgengenerate', message.author.id, message.channel, message.author, prompt, "", False))
            else:
                await message.channel.send("Queue limit has been reached, please wait for your previous gens to finish")


client = MetatronClient(intents=discord.Intents.all())  # client intents


@client.slash_command_tree.command()
async def summarize(interaction: discord.Interaction):
    """This is the slash command for summarize."""
    if not await client.is_enabled_not_banned("enableword", interaction.user):
        await interaction.response.send_message("LLM disabled or user banned", ephemeral=True, delete_after=5)
        return
    if await client.is_room_in_queue(interaction.user.id):
        await interaction.response.send_message("Summarizing...", ephemeral=True, delete_after=5)
        channel_history = [message async for message in interaction.channel.history(limit=40)]
        compiled_messages = '\n'.join([f'{msg.author}: {msg.content}' for msg in channel_history])
        prompt = f'Give a detailed summary of the following chat room conversation: {compiled_messages}'
        client.generation_queue_concurrency_list[interaction.user.id] += 1
        await client.generation_queue.put(('wordgensummary', interaction.user.id, interaction.channel, interaction.user, prompt))
    else:
        await interaction.response.send_message("Queue limit reached, please wait until your current gen or gens finish", ephemeral=True, delete_after=5)


@client.slash_command_tree.command()
async def wordgen(interaction: discord.Interaction, prompt: str, negative_prompt: Optional[str] = ""):
    """This is the slash command for wordgen"""
    if not await client.is_enabled_not_banned("enableword", interaction.user):
        await interaction.response.send_message("LLM disabled or user banned", ephemeral=True, delete_after=5)
        return
    if await client.is_room_in_queue(interaction.user.id):
        await interaction.response.send_message(f'PROMPT:{prompt}\nNEGATIVE:{negative_prompt}')
        client.generation_queue_concurrency_list[interaction.user.id] += 1
        await client.generation_queue.put(('wordgengenerate', interaction.user.id, interaction.channel, interaction.user, prompt, negative_prompt, False))
    else:
        await interaction.response.send_message("Queue limit reached, please wait until your current gen or gens finish", ephemeral=True, delete_after=5)


@client.slash_command_tree.command()
async def impersonate(interaction: discord.Interaction, userprompt: str, llmprompt: str):
    """This is the slash command for impersonate"""
    if not await client.is_enabled_not_banned("enableword", interaction.user):
        await interaction.response.send_message("LLM disabled or user banned", ephemeral=True, delete_after=5)
        return
    if await client.is_room_in_queue(interaction.user.id):
        client.generation_queue_concurrency_list[interaction.user.id] += 1
        await client.generation_queue.put(('wordgenimpersonate', interaction.user.id, userprompt, llmprompt, interaction.user.name))
        await interaction.response.send_message(f'History inserted:\n User: {userprompt}\n LLM: {llmprompt}')
    else:
        await interaction.response.send_message(
            "Queue limit reached, please wait until your current gen or gens finish")


@client.slash_command_tree.command()
@app_commands.choices(voicechoice=client.speak_voice_choices)
@app_commands.rename(userprompt='prompt', voicechoice='voice')
async def speakgen(interaction: discord.Interaction, userprompt: str, voicechoice: Optional[app_commands.Choice[str]] = None):
    """This is the slash command for speakgen."""
    if not await client.is_enabled_not_banned("enablespeak", interaction.user):
        await interaction.response.send_message("Bark disabled or user banned", ephemeral=True, delete_after=5)
        return
    if voicechoice is None:
        voiceselection = None
    else:
        voiceselection = voicechoice.name
    if await client.is_room_in_queue(interaction.user.id):
        await interaction.response.send_message("Generating Sound...", ephemeral=True, delete_after=5)
        client.generation_queue_concurrency_list[interaction.user.id] += 1
        await client.generation_queue.put(('speakgengenerate', interaction.user.id, userprompt, interaction.channel, voiceselection, interaction.user))
    else:
        await interaction.response.send_message("Queue limit reached, please wait until your current gen or gens finish")


@client.slash_command_tree.command()
@app_commands.choices(modelchoice=client.sd_model_choices)
@app_commands.choices(embeddingchoice=client.sd_embedding_choices)
@app_commands.choices(lorachoice=client.sd_loras_choices)
@app_commands.rename(userprompt='prompt', modelchoice='model', embeddingchoice='embedding', lorachoice='lora')
async def imagegen(interaction: discord.Interaction, userprompt: str, negativeprompt: Optional[str], modelchoice: Optional[app_commands.Choice[str]] = None, lorachoice: Optional[app_commands.Choice[str]] = None, embeddingchoice: Optional[app_commands.Choice[str]] = None, batch_size: Optional[int] = None, seed: Optional[int] = None, steps: Optional[int] = None, width: Optional[int] = None, height: Optional[int] = None, use_defaults: bool = True):
    """This is the slash command for imagegen."""
    if not await client.is_enabled_not_banned("enableimage", interaction.user):
        await interaction.response.send_message("SD disabled or user banned", ephemeral=True, delete_after=5)
        return
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
        client.generation_queue_concurrency_list[interaction.user.id] += 1
        await client.generation_queue.put(('imagegenerate', interaction.user.id, userprompt, interaction.channel, modelselection, batch_size, interaction.user.name, negativeprompt, seed, steps, width, height, use_defaults))
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
        loop.run_until_complete(quit_exit())  # If KeyboardInterrupt occurs during setup or start, perform cleanup and exit
    finally:
        loop.close()


if __name__ == "__main__":
    run_program()
