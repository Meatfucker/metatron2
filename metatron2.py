"""
metatron2 - A discord machine learning bot
"""
import os
os.environ["TQDM_DISABLE"] = "1"
import json
import io
import base64
import math
import re
import sys
import asyncio
from datetime import datetime
import discord
from discord import app_commands
from loguru import logger
from typing import Optional, Literal
from wordgen import load_llm, llm_generate, clear_history, Wordgenbuttons, delete_last_history, insert_history
from speakgen import load_bark, speak_generate, Speakgenbuttons, load_voices
from imagegen import load_sd, sd_generate, load_models_list, load_ti, load_embeddings_list
import torch
import gc

logger.remove()  # Remove the default configuration
SETTINGS = {}

with open("settings.cfg", "r", encoding="utf-8") as settings_file: #this builds the SETTINGS variable.
    for line in settings_file:
        if "=" in line:
            key, value = (line.split("=", 1)[0].strip(), line.split("=", 1)[1].strip())
            if key in SETTINGS:             # Check if the key already exists in SETTINGS
                if isinstance(SETTINGS[key], list):
                    SETTINGS[key].append(value)
                else: SETTINGS[key] = [SETTINGS[key], value]
            else: SETTINGS[key] = [value]  # Always store values as a list

if SETTINGS["enabledebug"][0] == "True": #this sets up the base logger formatting
    logger.add(sink=io.TextIOWrapper(sys.stdout.buffer, write_through=True), format="<light-black>{time:YYYY-MM-DD HH:mm:ss}</light-black> | <level>{level: <8}</level> | <cyan>{name: >8}</cyan>:<light-cyan>{function: <14}</light-cyan> | <light-yellow>{message: ^27}</light-yellow> | <light-red>{extra}</light-red>", level="DEBUG", colorize=True)
else:
    logger.add(sink=io.TextIOWrapper(sys.stdout.buffer, write_through=True), format="<light-black>{time:YYYY-MM-DD HH:mm:ss}</light-black> | <level>{level: <8}</level> | <light-yellow>{message: ^27}</light-yellow> | <light-red>{extra}</light-red>", level="INFO", colorize=True)

class MetatronClient(discord.Client):
    '''The discord client class for the bot'''
    def __init__(self, *, intents: discord.Intents):
        super().__init__(intents=intents)
        self.slash_command_tree = app_commands.CommandTree(self) #the object that holds the command tree for the slash commands
        self.llm_model = None #this holds the transformers llm model object
        self.llm_tokenizer = None #this holds the llm tokenizer object
        self.llm_view_last_message = {} #variable to track wordview buttons so there is only one set per user at a time
        self.generation_queue = asyncio.Queue() #the process queue object
        self.speak_voices_list = None #This is a list of available voice files
        self.speak_voice_choices = [] # The choices object for the discord speakgen ui
        self.sd_pipeline = None # This is the diffusers stable diffusion pipeline object
        self.sd_compel_processor = None #This is the diffusers compel processor object, which handles prompt weighting
        self.sd_model_list = None #This is a list of the available model files
        self.sd_model_choices = [] #The choices object for the discord imagegen models ui
        self.sd_loaded_embeddings = [] #The list of currently loaded image embeddings, needed to make diffusers not shit itself if you load the same one twice.
        self.sd_embeddings_list = None #List of available embeddings
        self.sd_embedding_choices = [] #The choices object for the discord imagegen embeddings ui
    
    async def setup_hook(self):
        if SETTINGS["enableword"][0] == "True":
            logger.info("Loading LLM")
            self.llm_model, self.llm_tokenizer = await load_llm() #load llm
        
        if SETTINGS["enablespeak"][0] == "True":
            logger.info("Loading Bark")
            await load_bark() #load the sound generation model
            logger.info("Loading Voices")
            self.speak_voices_list = await load_voices() #get the list of available voice files to build the discord interface with
            for voice in self.speak_voices_list:
                self.speak_voice_choices.append(app_commands.Choice(name=voice, value=voice))
        
        if SETTINGS["enableimage"][0] == "True":
            logger.info("Loading SD")
            self.sd_pipeline, self.sd_compel_processor = await load_sd() #load the sd model pipeline and compel prompt processor
            self.sd_model_list = await load_models_list() #get the list of available models to build the discord interface with
            for model in self.sd_model_list:
                self.sd_model_choices.append(app_commands.Choice(name=model, value=model))
            self.sd_embeddings_list = await load_embeddings_list() #get the list of available embeddings to biuld the discord interface with
            for embedding in self.sd_embeddings_list:
                self.sd_embedding_choices.append(app_commands.Choice(name=embedding, value=embedding))
                        
        self.loop.create_task(client.process_queue()) #start queue
        await self.slash_command_tree.sync() #sync commands to discord
        logger.info("Logging in...")
    
    async def on_ready(self):
        self.ready_logger = logger.bind(user=client.user.name, userid=client.user.id)
        self.ready_logger.info("Login Successful")
    
    async def process_queue(self):
        while True:
            args = await self.generation_queue.get()
            action = args[0] #first argument passed to queue should always be the action to do
            if SETTINGS["enableword"][0] == "True":
                if action == 'wordgenforget':
                    message = args[1]
                    await clear_history(message)
                    llm_clear_history_logger = logger.bind(user=message.author.name, userid=message.author.id)
                    llm_clear_history_logger.success("WORDGEN History Cleared.")
                elif action == 'wordgengenerate':
                    message, stripped_message = args[1:3]
                    response = await llm_generate(message, stripped_message, self.llm_model, self.llm_tokenizer, SETTINGS["wordsystemprompt"][0], SETTINGS["wordnegprompt"][0]) #generate the text
                    if message.author.id in self.llm_view_last_message: #check if there are an existing set of llm buttons for the user and if so, delete them
                        try:
                            await self.llm_view_last_message[message.author.id].delete()
                        except discord.NotFound:
                            pass  # Message not found, might have been deleted already
                    chunks = [response[i:i+1500] for i in range(0, len(response), 1500)] #split the reply into 1500 char pieces so as not to bump up against the discord message length limit
                    for chunk in chunks:
                        await message.channel.send(chunk)
                    new_message = await message.channel.send(view=Wordgenbuttons(self.generation_queue, message.author.id, message, stripped_message)) #send the message with the llm buttons
                    self.llm_view_last_message[message.author.id] = new_message #track the message id of the last set of llm buttons for each user
                    self.llm_reply_logger = logger.bind(user=message.author.name, userid=message.author.id, prompt=stripped_message, reply=response)
                    self.llm_reply_logger.success("WORDGEN Reply")
                    with torch.no_grad(): #clear torch gpu cache, freeing up vram
                        torch.cuda.empty_cache()
                    gc.collect() #clear python garbage, freeing up ram (and maybe vram?)
                elif action == 'wordgendeletelast':
                    await delete_last_history(message)
                    self.llm_delete_last_logger=logger.bind(user=message.author.name, userid=message.author.id)
                    self.llm_delete_last_logger.success("WORDGEN Reply Deleted.")
                elif action == 'wordgenimpersonate':
                    userid, prompt, llmprompt, username = args[1:5]
                    await insert_history(userid, prompt, llmprompt, SETTINGS["wordsystemprompt"][0])
                    self.wordgenreply_logger = logger.bind(user=username, userid=userid, prompt=prompt, llmprompt=llmprompt)
                    self.wordgenreply_logger.success("WORDGEN Impersonate.")
            if SETTINGS["enablespeak"][0] == "True":
                if action == 'speakgengenerate':
                    prompt, channel, userid, voicefile, username = args[1:6]
                    wav_bytes_io = await speak_generate(prompt, voicefile) #this generates the audio
                    truncatedprompt = prompt[:1000] #this avoids file name length limits
                    await channel.send(content=f"Prompt:`{prompt}`", file=discord.File(wav_bytes_io, filename=f"{truncatedprompt}.wav"), view=Speakgenbuttons(self.generation_queue, userid, prompt, voicefile))
                    self.speakgenreply_logger = logger.bind(user=username, userid=userid, prompt=prompt)
                    self.speakgenreply_logger.success("SPEAKGEN Replied")
                    with torch.no_grad(): #clear gpu memory cache
                        torch.cuda.empty_cache()
                    gc.collect() #clear python memory
            if SETTINGS["enableimage"][0] == "True":
                if action == 'imagegenerate':
                    prompt, channel, sdmodel, batch_size, username, userid = args[1:7]
                    if sdmodel != None: #if a model has been selected, create and load a fresh pipeline and compel processor
                        self.sd_pipeline, self.sd_compel_processor = await load_sd(sdmodel, self.sd_pipeline)
                        self.sd_loaded_embeddings = []
                    self.sd_pipeline, loaded_image_embeddings = await load_ti(self.sd_pipeline, prompt, self.sd_loaded_embeddings) #check for embeddings and apply them
                    generatedimage = await sd_generate(self.sd_pipeline, self.sd_compel_processor, prompt, sdmodel, batch_size) #generate the image request
                    truncatedprompt = prompt[:1000] #this avoids file name length issues
                    await channel.send(content=f"Prompt:`{prompt}`", file=discord.File(generatedimage, filename=f"{truncatedprompt}.png"))
                    self.imagegenreply_logger = logger.bind(user=username, userid=userid, prompt=prompt, model=sdmodel, batchsize=batch_size)
                    self.imagegenreply_logger.success("IMAGEGEN Replied")
                    with torch.no_grad(): #clear gpu memory cache
                        torch.cuda.empty_cache()
                    gc.collect() #clear python memory
            else:
                logger.error(f'QUEUE Error, Unknown function:{action}:{args}')
            
    async def on_message(self, message):
        if self.user.mentioned_in(message):
            if SETTINGS["enableword"][0] != "True":
                await message.channel.send("LLM generation is currently disabled.")
                return #check if LLM generation is enabled
            stripped_message = re.sub(r'<[^>]+>', '', message.content).lstrip() #this removes the user tag
            if "forget" in stripped_message.lower():
                await self.generation_queue.put(('wordgenforget', message))
            else:
                await self.generation_queue.put(('wordgengenerate', message, stripped_message))
        #await self.process_commands(message)
   
client = MetatronClient(intents=discord.Intents.all()) #client intents

@client.slash_command_tree.command()
async def impersonate(interaction: discord.Interaction, userprompt: str, llmprompt: str):
    '''Slash command that allows for one shot prompting'''
    if SETTINGS["enableword"][0] != "True":
                await interaction.response.send_message("LLM generation is currently disabled.")
                return #check if LLM generation is enabled
    await client.generation_queue.put(('wordgenimpersonate', interaction.user.id, userprompt, llmprompt, interaction.user.name))
    await interaction.response.send_message(f'History inserted:\n User: {userprompt}\n LLM: {llmprompt}')
    
@client.slash_command_tree.command()
@app_commands.choices(voicechoice=client.speak_voice_choices)
async def speakgen(interaction: discord.Interaction, userprompt: str, voicechoice: Optional[app_commands.Choice[str]] = None):
    '''Slash command that generates speech'''
    if SETTINGS["enablespeak"][0] != "True":
                await interaction.response.send_message("Sound generation is currently disabled.")
                return #check if sound generation is enabled
    if voicechoice == None:
        voiceselection = None
    else:
        voiceselection = voicechoice.name
    await interaction.response.send_message("Generating Sound...", ephemeral=True, delete_after=5)
    await client.generation_queue.put(('speakgengenerate', userprompt, interaction.channel, interaction.user.id, voiceselection, interaction.user.name))

@client.slash_command_tree.command()
@app_commands.choices(modelchoice=client.sd_model_choices)
@app_commands.choices(embeddingchoice=client.sd_embedding_choices)
async def imagegen(interaction: discord.Interaction, userprompt: str, modelchoice: Optional[app_commands.Choice[str]] = None, batch_size: Optional[int] = 1, embeddingchoice: Optional[app_commands.Choice[str]] = None):
    '''Slash command that generates images'''
    if SETTINGS["enableimage"][0] != "True":
                await interaction.response.send_message("Image generation is currently disabled.")
                return #check if image generation is enabled
    if modelchoice == None:
        modelselection = None
    else:
        modelselection = modelchoice.name
    if embeddingchoice != None:
        userprompt = f"{userprompt}{embeddingchoice.name}"
    await interaction.response.send_message("Generating Image...", ephemeral=True, delete_after=5)
    await client.generation_queue.put(('imagegenerate', userprompt, interaction.channel, modelselection, batch_size, interaction.user.name, interaction.user.id))

client.run(SETTINGS["token"][0], log_handler=None) #run bot