"""
metatron2 - A discord machine learning bot
"""
from typing import Optional
import json
import io
import base64
import math
import re
from datetime import datetime
import discord
from discord import app_commands
from wordgen import load_llm, wordgen, clearhistory
import asyncio

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

    async def setup_hook(self): 
        await self.tree.sync()
        if SETTINGS["enableword"][0] == "True":
            self.wordgenmodel, self.wordgentokenizer = await load_llm()
        self.loop.create_task(client.process_wordgen_queue())

    async def on_ready(self):
        print("Logged in")
        
    async def process_wordgen_queue(self):
        while True:
            (message, stripped_message, kwargs) = await self.wordgen_queue.get()

            if kwargs.get('forget', True):
                await clearhistory(message)
            else:
                response = await wordgen(message, stripped_message, self.wordgenmodel, self.wordgentokenizer, SETTINGS["wordsystemprompt"][0], SETTINGS["wordnegprompt"][0])
                chunks = [response[i:i+1500] for i in range(0, len(response), 1500)]
                for chunk in chunks:
                    await message.channel.send(chunk)
            self.wordgen_queue.task_done()

    async def on_message(self, message):
        if self.user.mentioned_in(message):
            if SETTINGS["enableword"][0] != "True":
                await message.channel.send("LLM generation is currently disabled.")
                return #check if LLM generation is enabled
            stripped_message = re.sub(r'<[^>]+>', '', message.content).lstrip()
            if "forget" in stripped_message.lower():
                await self.wordgen_queue.put((message, stripped_message, {'forget': True}))
            else:
                await self.wordgen_queue.put((message, stripped_message, {'forget': False}))
   
client = MetatronClient(intents=discord.Intents.all()) #client intents
client.run(SETTINGS["token"][0]) #run bot