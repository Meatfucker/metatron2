#speakgen.py - Functions for Bark capabilities
import os
os.environ["TQDM_DISABLE"] = "1"
import io
import asyncio
import torch
from loguru import logger
from scipy.io.wavfile import write as write_wav
import discord
from discord import app_commands
from bark import SAMPLE_RATE, generate_audio, preload_models

logger.remove()

async def load_bark():
    preload_models()
    logger.success("Bark Model Loaded.")
    
async def load_voices():
    """Get list of voices for user interface"""
    voices = []
    voices_list = os.listdir("voices/")
    for voice_file in voices_list:
        voices.append(voice_file)
    return voices

async def speakgenerate(prompt, voicefile):
    """Function to generate speech"""
    logger.info("SPEAKGEN Generate Started")
    if voicefile != None:
        voicechoice = f'voices/{voicefile}'
        audio_array = await asyncio.to_thread(generate_audio, prompt, history_prompt=voicechoice)
    else:
        audio_array = await asyncio.to_thread(generate_audio, prompt)
    logger.success("SPEAKGEN Generate Completed")
    wav_io = io.BytesIO()
    write_wav(wav_io, SAMPLE_RATE, audio_array)
    wav_io.seek(0)
    return wav_io

class Speakgenbuttons(discord.ui.View):
    """Class for the ui buttons on speakgen"""

    def __init__(self, generation_queue, userid, prompt, voicefile):
        super().__init__()
        self.timeout = None
        self.generation_queue = generation_queue
        self.userid = userid
        self.prompt = prompt
        self.voicefile = voicefile
    
    @discord.ui.button(label='Reroll', emoji="🎲", style=discord.ButtonStyle.grey)
    async def reroll(self, interaction: discord.Interaction, button: discord.ui.Button):
        """Rerolls last reply"""
        if self.userid == interaction.user.id:
            await interaction.response.send_message("Rerolling...", ephemeral=True, delete_after=5)
            await self.generation_queue.put(('speakgengenerate', self.prompt, interaction.channel, self.userid, self.voicefile))
    
    @discord.ui.button(label='Mail', emoji="✉", style=discord.ButtonStyle.grey)
    async def dmimage(self, interaction: discord.Interaction, button: discord.ui.Button):
        """DMs sound"""
        await interaction.response.send_message("DM'ing sound...", ephemeral=True, delete_after=5)
        sound_bytes = await interaction.message.attachments[0].read()
        dm_channel = await interaction.user.create_dm()
        truncatedfilename = self.prompt[:1000]
        await dm_channel.send(file=discord.File(io.BytesIO(sound_bytes), filename=f'{truncatedfilename}.wav'))
        logger.success("SPEAKGEN DM successful")

    @discord.ui.button(label='Delete', emoji="❌", style=discord.ButtonStyle.grey)
    async def delete_message(self, interaction: discord.Interaction, button: discord.ui.Button):
        """Deletes message"""
        if self.userid == interaction.user.id:
            await interaction.message.delete()
        await interaction.response.send_message("Sound deleted.", ephemeral=True, delete_after=5)
        logger.info("SPEAKGEN Delete")
