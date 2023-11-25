# speakgen.py - Functions for Bark capabilities
import os
import io
import asyncio
import gc
import re
from datetime import datetime
from loguru import logger
from scipy.io.wavfile import write as write_wav
import discord
from bark import SAMPLE_RATE, generate_audio, preload_models
from pydub import AudioSegment
from modules.settings import SETTINGS


async def load_bark():
    """This loads the bark models"""
    preload_models()
    logger.success("Bark Loaded.")


async def load_voices():
    """Get list of voices for user interface"""
    voices = []
    voices_list = os.listdir("models/voices/")
    for voice_file in voices_list:
        if voice_file.endswith(".npz"):
            voices.append(voice_file)
    return voices


class VoiceQueueObject:

    def __init__(self, action, metatron, user, channel, prompt, voice_file=None, user_voice_file=None):
        self.action = action  # This is the generation queue action
        self.metatron = metatron  # This is the discord client
        self.user = user  # This is the discord variable that contains user.name and user.id
        self.channel = channel  # This is the discord variable for the channel, includes the functions to send messages, etc
        self.prompt = prompt  # This is the users prompt
        if voice_file is not None:  # This holds the voice file selection
            self.voice_file = voice_file.name
        else:
            self.voice_file = voice_file
        self.user_voice_file = user_voice_file
        self.audio = None  # This holds the audio after generation
        self.sanitized_prompt = re.sub(r'[^\w\s\-.]', '', self.prompt)[:100]  # This contains a prompt thats safe to use as a filename

    async def generate(self):
        """Generates audio"""
        speakgen_logger = logger.bind(prompt=self.prompt)  # Bind useful info to logurus extras dict
        speakgen_logger.info("SPEAKGEN Generate started")
        if self.user_voice_file is not None:
            await self.user_voice_file.save("outputs/voice.npz")
            audio_array = await asyncio.to_thread(generate_audio, self.prompt, "outputs/voice.npz", silent=True)
            os.remove("outputs/voice.npz")
        else:
            if self.voice_file is not None:  # If there is a voice file, include in the generation call.
                voice_path = f'models/voices/{self.voice_file}'
                audio_array = await asyncio.to_thread(generate_audio, self.prompt, voice_path, silent=True)
            else:
                audio_array = await asyncio.to_thread(generate_audio, self.prompt, silent=True)

        speakgen_logger.debug("SPEAKGEN Generate finished")

        wav_io = io.BytesIO()  # Create a file like object for the audio
        write_wav(wav_io, SAMPLE_RATE, audio_array)  # Put the audio in it.
        wav_io.seek(0)  # Return to the beginning of the file object
        if SETTINGS["saveinmp3"][0] == "True":
            audio_segment = AudioSegment.from_file(wav_io, format="wav")  # Load the WAV data into an AudioSegment
            mp3_io = io.BytesIO()
            audio_segment.export(mp3_io, format="mp3")  # Export the audio as an MP3 file-like object
            mp3_io.seek(0)
            self.audio = mp3_io
        else:
            self.audio = wav_io
        gc.collect()

    async def save(self):
        """Saves audio to disk"""
        current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if SETTINGS["saveinmp3"][0] == "True":
            savepath = f'{SETTINGS["savepath"][0]}/{current_datetime}-{self.sanitized_prompt}.mp3'
        else:
            savepath = f'{SETTINGS["savepath"][0]}/{current_datetime}-{self.sanitized_prompt}.wav'
        with open(savepath, "wb") as output_file:
            output_file.write(self.audio.getvalue())

    async def respond(self):
        if SETTINGS["saveinmp3"][0] == "True":
            await self.channel.send(content=f"Prompt:`{self.prompt}`", file=discord.File(self.audio, filename=f"{self.sanitized_prompt}.mp3"), view=Speakgenbuttons(self))
        else:
            await self.channel.send(content=f"Prompt:`{self.prompt}`", file=discord.File(self.audio, filename=f"{self.sanitized_prompt}.wav"), view=Speakgenbuttons(self))
        speakgenreply_logger = logger.bind(user=self.user.name, prompt=self.prompt)
        speakgenreply_logger.success("SPEAKGEN Replied")


class Speakgenbuttons(discord.ui.View):
    """Class for the ui buttons on speakgen"""
    def __init__(self, voiceobject):
        super().__init__()
        self.timeout = None  # Disables the timeout on the buttons
        self.voiceobject = voiceobject

    @discord.ui.button(label='Reroll', emoji="🎲", style=discord.ButtonStyle.grey)
    async def reroll(self, interaction: discord.Interaction, button: discord.ui.Button):
        """Rerolls last reply"""
        if self.voiceobject.user.id == interaction.user.id:
            if await self.voiceobject.metatron.is_room_in_queue(self.voiceobject.user.id):
                await interaction.response.send_message("Rerolling...", ephemeral=True, delete_after=5)
                self.voiceobject.metatron.generation_queue_concurrency_list[self.voiceobject.user.id] += 1
                await self.voiceobject.metatron.generation_queue.put(self.voiceobject)
            else:
                await interaction.response.send_message("Queue limit reached, please wait until your current gen or gens finish")

    @discord.ui.button(label='Mail', emoji="✉", style=discord.ButtonStyle.grey)
    async def dm_sound(self, interaction: discord.Interaction, button: discord.ui.Button):
        """DMs sound"""
        await interaction.response.send_message("DM'ing sound...", ephemeral=True, delete_after=5)
        sound_bytes = await interaction.message.attachments[0].read()
        dm_channel = await interaction.user.create_dm()
        await dm_channel.send(file=discord.File(io.BytesIO(sound_bytes), filename=f'{self.voiceobject.sanitized_prompt}.wav'))
        speak_dm_logger = logger.bind(user=interaction.user.name, userid=interaction.user.id)
        speak_dm_logger.success("SPEAKGEN DM successful")

    @discord.ui.button(label='Delete', emoji="❌", style=discord.ButtonStyle.grey)
    async def delete_message(self, interaction: discord.Interaction, button: discord.ui.Button):
        """Deletes message"""
        if self.voiceobject.user.id == interaction.user.id:
            await interaction.message.delete()
        await interaction.response.send_message("Sound deleted.", ephemeral=True, delete_after=5)
        speak_delete_logger = logger.bind(user=interaction.user.name, userid=interaction.user.id)
        speak_delete_logger.info("SPEAKGEN Delete")
