"""
metatron2 - A discord machine learning bot
"""
from typing import Optional
import json
import io
import base64
import math
import re
import sys
from datetime import datetime
import discord
from discord import app_commands
from wordgen import load_llm, wordgen, clearhistory, Wordgenbuttons, deletelasthistory, inserthistory
import asyncio
from loguru import logger

logger.remove()  # Remove the default configuration
logger.add(sink=io.TextIOWrapper(sys.stdout.buffer, write_through=True), format="<light-black>{time:YYYY-MM-DD HH:mm:ss}</light-black> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<light-cyan>{function}</light-cyan> | <light-yellow>{message}</light-yellow>", level="INFO", colorize=True)

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
  
class MetatronClient(discord.Client):
    
    def __init__(self, *, intents: discord.Intents):
        super().__init__(intents=intents)
        self.tree = app_commands.CommandTree(self)
        self.wordgenmodel = None
        self.wordgentokenizer = None
        self.wordgen_queue = asyncio.Queue()
        self.wordgenview_lastmessage = {}

    async def setup_hook(self):
        logger.info("Logging in...")
        await self.tree.sync()
        if SETTINGS["enableword"][0] == "True":
            self.wordgenmodel, self.wordgentokenizer = await load_llm()
        self.loop.create_task(client.process_queue())

    async def on_ready(self):
        logger.info("Login Successful")
    
    async def process_queue(self):
        while True:
            args = await self.wordgen_queue.get()
            action = args[0]
            if action == 'wordgenforget':
                message = args[1]
                await clearhistory(message)
                logger.success(f'WORDGEN History cleared.')
            elif action == 'wordgengenerate':
                message = args[1]
                stripped_message = args[2]
                response = await wordgen(args[1], args[2], self.wordgenmodel, self.wordgentokenizer, SETTINGS["wordsystemprompt"][0], SETTINGS["wordnegprompt"][0])
                if message.author.id in self.wordgenview_lastmessage:
                    try:
                        await self.wordgenview_lastmessage[message.author.id].delete()
                    except discord.NotFound:
                        pass  # Message not found, might have been deleted already
                chunks = [response[i:i+1500] for i in range(0, len(response), 1500)]
                for chunk in chunks:
                    await message.channel.send(chunk)
                new_message = await message.channel.send(view=Wordgenbuttons(self.wordgen_queue, message.author.id, message, stripped_message))
                self.wordgenview_lastmessage[message.author.id] = new_message
                logger.success(f'WORDGEN Reply:{stripped_message}')
            elif action == 'wordgendeletelast':
                await deletelasthistory(message)
                logger.success(f'WORDGEN Last History Pair Delete')
            elif action == 'wordgenimpersonate':
                userid = args[1]
                prompt = args[2]
                llmprompt = args[3]
                await inserthistory(userid, prompt, llmprompt, SETTINGS["wordsystemprompt"][0])
            else:
                logger.error(f'QUEUE Error, Unknown function:{action}:{args}')
            
    async def on_message(self, message):
        if self.user.mentioned_in(message):
            if SETTINGS["enableword"][0] != "True":
                await message.channel.send("LLM generation is currently disabled.")
                return #check if LLM generation is enabled
            stripped_message = re.sub(r'<[^>]+>', '', message.content).lstrip()
            if "forget" in stripped_message.lower():
                await self.wordgen_queue.put(('wordgenforget', message))
            else:
                await self.wordgen_queue.put(('wordgengenerate', message, stripped_message))
        await self.process_commands(message)
   
client = MetatronClient(intents=discord.Intents.all()) #client intents

@client.tree.command()
async def impersonate(interaction: discord.Interaction, userprompt: str, llmprompt: str):
    """Slash command that allows for one shot prompting"""
    await client.wordgen_queue.put(('wordgenimpersonate', interaction.user.id, userprompt, llmprompt))
    await interaction.response.send_message(f'History inserted:\n User: {userprompt}\n LLM: {llmprompt}')
    
client.run(SETTINGS["token"][0], log_handler=None) #run bot


   