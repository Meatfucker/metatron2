#speakgen.py - Functions for Bark capabilities
import os
os.environ["TQDM_DISABLE"] = "1" #Attempt to disable annoying tqdm progress bars
import io
import asyncio
import torch
from loguru import logger
from scipy.io.wavfile import write as write_wav
import discord
from discord import app_commands
from bark import SAMPLE_RATE, generate_audio, preload_models
from modules.settings import SETTINGS

logger.remove() #attempt to silence noisy library logging messages

@logger.catch
async def load_bark():
    '''This loads the bark models'''
    preload_models()
    logger.success("Bark Model Loaded.")

@logger.catch    
async def load_voices():
    '''Get list of voices for user interface'''
    voices = []
    voices_list = os.listdir("voices/")
    for voice_file in voices_list:
        if voice_file.endswith(".npz"):
            voices.append(voice_file)
    return voices

@logger.catch
async def speak_generate(prompt, voice_file):
    '''Function to generate speech'''
    logger.debug("SPEAKGEN Generate Started")
    if voice_file != None:
        voice_choice = f'voices/{voice_file}'
        audio_array = await asyncio.to_thread(generate_audio, prompt, history_prompt=voice_choice) #Thread the generate call so it doesnt lock up the bot client.
    else:
        audio_array = await asyncio.to_thread(generate_audio, prompt) #Thread the generate call so it doesnt lock up the bot client.
    logger.debug("SPEAKGEN Generate Completed")
    wav_io = io.BytesIO()
    write_wav(wav_io, SAMPLE_RATE, audio_array) #turn the generated audio into a wav file-like object
    wav_io.seek(0)
    return wav_io

class Speakgenbuttons(discord.ui.View):
    '''Class for the ui buttons on speakgen'''

    def __init__(self, generation_queue, userid, prompt, voice_file, metatron_client):
        super().__init__()
        self.timeout = None #Disables the timeout on the buttons
        self.generation_queue = generation_queue
        self.userid = userid
        self.prompt = prompt
        self.voice_file = voice_file
        self.metatron_client = metatron_client
    
    @logger.catch
    @discord.ui.button(label='Reroll', emoji="🎲", style=discord.ButtonStyle.grey)
    async def reroll(self, interaction: discord.Interaction, button: discord.ui.Button):
        '''Rerolls last reply'''
        if self.userid == interaction.user.id:
            if await self.metatron_client.is_room_in_queue(self.userid) == True:
                await interaction.response.send_message("Rerolling...", ephemeral=True, delete_after=5)
                await self.generation_queue.put(('speakgengenerate', self.userid, self.prompt, interaction.channel, self.voice_file, interaction.user.name))
            else:
                await interaction.response.send_message("Queue limit reached, please wait until your current gen or gens finish")
    
    @logger.catch
    @discord.ui.button(label='Mail', emoji="✉", style=discord.ButtonStyle.grey)
    async def dmimage(self, interaction: discord.Interaction, button: discord.ui.Button):
        '''DMs sound'''
        await interaction.response.send_message("DM'ing sound...", ephemeral=True, delete_after=5)
        sound_bytes = await interaction.message.attachments[0].read()
        dm_channel = await interaction.user.create_dm()
        truncated_filename = self.prompt[:1000]
        await dm_channel.send(file=discord.File(io.BytesIO(sound_bytes), filename=f'{truncated_filename}.wav'))
        speak_dm_logger = logger.bind(user=interaction.user.name, userid=interaction.user.id)
        speak_dm_logger.success("SPEAKGEN DM successful")

    @logger.catch
    @discord.ui.button(label='Delete', emoji="❌", style=discord.ButtonStyle.grey)
    async def delete_message(self, interaction: discord.Interaction, button: discord.ui.Button):
        '''Deletes message'''
        if self.userid == interaction.user.id:
            await interaction.message.delete()
        await interaction.response.send_message("Sound deleted.", ephemeral=True, delete_after=5)
        speak_delete_logger = logger.bind(user=interaction.user.name, userid=interaction.user.id)
        speak_delete_logger.info("SPEAKGEN Delete")