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
from wordgen import load_llm, wordgen, clearhistory, Wordgenbuttons, deletelasthistory, inserthistory
from speakgen import load_bark, speakgenerate, Speakgenbuttons, load_voices
from imagegen import load_sd, imagegenerate

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
if SETTINGS["enabledebug"][0] == "True":
    logger.add(sink=io.TextIOWrapper(sys.stdout.buffer, write_through=True), format="<light-black>{time:YYYY-MM-DD HH:mm:ss}</light-black> | <level>{level: <8}</level> | <cyan>{name: >10}</cyan>:<light-cyan>{function: <14}</light-cyan> | <light-yellow>{message: ^27}</light-yellow> | <light-red>{extra}</light-red>", level="DEBUG", colorize=True)
else:
    logger.add(sink=io.TextIOWrapper(sys.stdout.buffer, write_through=True), format="<light-black>{time:YYYY-MM-DD HH:mm:ss}</light-black> | <level>{level: <8}</level> | <cyan>{name: >10}</cyan>:<light-cyan>{function: <14}</light-cyan> | <light-yellow>{message: ^27}</light-yellow> | <light-red>{extra}</light-red>", level="INFO", colorize=True)

class MetatronClient(discord.Client):
    
    def __init__(self, *, intents: discord.Intents):
        super().__init__(intents=intents)
        self.tree = app_commands.CommandTree(self)
        self.wordgenmodel = None
        self.wordgentokenizer = None
        self.generation_queue = asyncio.Queue() #create the queue object
        self.wordgenview_lastmessage = {} #variable to track wordview buttons so there is only one set per user at a time
        self.speakgenvoices = None
        self.voice_choices = []
        self.imagepipeline = None
        self.compel_proc = None
    
    async def setup_hook(self):
        if SETTINGS["enableword"][0] == "True":
            logger.info("Loading LLM")
            self.wordgenmodel, self.wordgentokenizer = await load_llm() #load llm
        
        if SETTINGS["enablespeak"][0] == "True":
            logger.info("Loading Bark")
            await load_bark()
            logger.info("Loading Voices")
            self.speakgenvoices = await load_voices()
            for voice in self.speakgenvoices:
                self.voice_choices.append(app_commands.Choice(name=voice, value=voice))
        
        if SETTINGS["enableimage"][0] == "True":
            logger.info("Loading SD")
            self.imagepipeline, self.compel_proc = await load_sd()
                        
        self.loop.create_task(client.process_queue()) #start queue
        await self.tree.sync() #sync commands to discord
        logger.info("Logging in...")
    
    async def on_ready(self):
        self.ready_logger = logger.bind(user=client.user.name, userid=client.user.id)
        self.ready_logger.info("Login Successful")
    
    async def process_queue(self):
        while True:
            args = await self.generation_queue.get()
            action = args[0]
            if SETTINGS["enableword"][0] == "True":
                if action == 'wordgenforget':
                    message = args[1]
                    await clearhistory(message)
                    logger.success("WORDGEN History Cleared.")
                elif action == 'wordgengenerate':
                    message, stripped_message = args[1:3]
                    response = await wordgen(message, stripped_message, self.wordgenmodel, self.wordgentokenizer, SETTINGS["wordsystemprompt"][0], SETTINGS["wordnegprompt"][0])
                    if message.author.id in self.wordgenview_lastmessage:
                        try:
                            await self.wordgenview_lastmessage[message.author.id].delete()
                        except discord.NotFound:
                            pass  # Message not found, might have been deleted already
                    chunks = [response[i:i+1500] for i in range(0, len(response), 1500)]
                    for chunk in chunks:
                        await message.channel.send(chunk)
                    new_message = await message.channel.send(view=Wordgenbuttons(self.generation_queue, message.author.id, message, stripped_message))
                    self.wordgenview_lastmessage[message.author.id] = new_message
                    self.wordgenreply_logger = logger.bind(prompt=stripped_message, reply=response)
                    self.wordgenreply_logger.success("WORDGEN Reply")
                elif action == 'wordgendeletelast':
                    await deletelasthistory(message)
                    logger.success("WORDGEN Reply Deleted.")
                elif action == 'wordgenimpersonate':
                    userid, prompt, llmprompt = args[1:4]
                    await inserthistory(userid, prompt, llmprompt, SETTINGS["wordsystemprompt"][0])
                    self.wordgenreply_logger = logger.bind(prompt=prompt, llmprompt=llmprompt)
                    self.wordgenreply_logger.success("WORDGEN Impersonate.")
            if SETTINGS["enablespeak"][0] == "True":
                if action == 'speakgengenerate':
                    prompt, channel, userid, voicefile = args[1:5]
                    wav_bytes_io = await speakgenerate(prompt, voicefile)
                    logger.success("SPEAKGEN Audio Replied.")
                    truncatedprompt = prompt[:1000]
                    await channel.send(content=f"Prompt:`{prompt}`", file=discord.File(wav_bytes_io, filename=f"{truncatedprompt}.wav"), view=Speakgenbuttons(self.generation_queue, userid, prompt, voicefile))
                    self.speakgenreply_logger = logger.bind(prompt=prompt)
                    self.speakgenreply_logger.success("SPEAKGEN Replied")
            if SETTINGS["enableimage"][0] == "True":
                if action == 'imagegenerate':
                    prompt, channel, batch_size = args[1:4]
                    generatedimage = await imagegenerate(self.imagepipeline, self.compel_proc, prompt, batch_size)
                    truncatedprompt = prompt[:1000]
                    await channel.send(content=f"Prompt:`{prompt}`", file=discord.File(generatedimage, filename=f"{truncatedprompt}.png"))
                    self.imagegenreply_logger = logger.bind(prompt=prompt)
                    self.imagegenreply_logger.success("IMAGEGEN Replied")
            
            else:
                logger.error(f'QUEUE Error, Unknown function:{action}:{args}')
            
    async def on_message(self, message):
        if self.user.mentioned_in(message):
            if SETTINGS["enableword"][0] != "True":
                await message.channel.send("LLM generation is currently disabled.")
                return #check if LLM generation is enabled
            stripped_message = re.sub(r'<[^>]+>', '', message.content).lstrip()
            if "forget" in stripped_message.lower():
                await self.generation_queue.put(('wordgenforget', message))
            else:
                await self.generation_queue.put(('wordgengenerate', message, stripped_message))
        await self.process_commands(message)
   
client = MetatronClient(intents=discord.Intents.all()) #client intents

@client.tree.command()
async def impersonate(interaction: discord.Interaction, userprompt: str, llmprompt: str):
    """Slash command that allows for one shot prompting"""
    if SETTINGS["enableword"][0] != "True":
                await interaction.response.send_message("LLM generation is currently disabled.")
                return #check if LLM generation is enabled
    await client.generation_queue.put(('wordgenimpersonate', interaction.user.id, userprompt, llmprompt))
    await interaction.response.send_message(f'History inserted:\n User: {userprompt}\n LLM: {llmprompt}')
    
@client.tree.command()
@app_commands.choices(voicechoice=client.voice_choices)
async def speakgen(interaction: discord.Interaction, userprompt: str, voicechoice: Optional[app_commands.Choice[str]] = None):
    """Slash command that generates speech"""
    if SETTINGS["enablespeak"][0] != "True":
                await interaction.response.send_message("Sound generation is currently disabled.")
                return #check if sound generation is enabled
    if voicechoice == None:
        voiceselection = None
    else:
        voiceselection = voicechoice.name
    await client.generation_queue.put(('speakgengenerate', userprompt, interaction.channel, interaction.user.id, voiceselection))
    await interaction.response.send_message("Generating speakgen...", ephemeral=True, delete_after=5)
    
@client.tree.command()
async def imagegen(interaction: discord.Interaction, userprompt: str, batch_size: Optional[int] = 1):
    """Slash command that generates images"""
    if SETTINGS["enableimage"][0] != "True":
                await interaction.response.send_message("Image generation is currently disabled.")
                return #check if image generation is enabled
    await client.generation_queue.put(('imagegenerate', userprompt, interaction.channel, batch_size))
    await interaction.response.send_message("Generating speakgen...", ephemeral=True, delete_after=5)

client.run(SETTINGS["token"][0], log_handler=None) #run bot