import numpy as np
import torch
import torchaudio
import io
import asyncio
from loguru import logger
from pydub import AudioSegment
from encodec import EncodecModel
from encodec.utils import convert_audio
from modules.bark_hubert_quantizer.hubert_manager import HuBERTManager
from modules.bark_hubert_quantizer.pre_kmeans_hubert import CustomHubert
from modules.bark_hubert_quantizer.customtokenizer import CustomTokenizer
from modules.settings import SETTINGS
import discord

class CloneQueueObject:
        @logger.catch()
        def __init__(self, action, metatron, user, channel, input_audio, file_name):

            self.action = action
            self.metatron = metatron
            self.user = user
            self.channel = channel
            self.input_audio = input_audio
            self.file_name = file_name
            self.speaker_file = io.BytesIO()
            self.audio_type = None


        @logger.catch()
        async def clone_voice(self):

            logger.debug("Loading Hubert")
            hubert_installed = await asyncio.to_thread(HuBERTManager.make_sure_hubert_installed)
            hubert_model = CustomHubert(hubert_installed, device="cpu")
            tokenizer_installed = await asyncio.to_thread(HuBERTManager.make_sure_tokenizer_installed, model="quantifier_V1_hubert_base_ls960_23.pth", local_file="tokenizer_large.pth")
            quant_model = CustomTokenizer.load_from_checkpoint(tokenizer_installed, "cpu")
            encodec_model = EncodecModel.encodec_model_24khz()
            encodec_model.set_target_bandwidth(6.0)
            encodec_model.to("cpu")
            self.input_audio.seek(0)
            wav, sr = torchaudio.load(self.input_audio)
            wav_hubert = wav.to("cpu")
            if wav_hubert.shape[0] == 2:  # Stereo to mono if needed
                wav_hubert = wav_hubert.mean(0, keepdim=True)
            semantic_vectors = hubert_model.forward(wav_hubert, input_sample_hz=sr)
            semantic_tokens = quant_model.get_token(semantic_vectors)
            wav = convert_audio(wav, sr, encodec_model.sample_rate, 1).unsqueeze(0)
            wav = wav.to("cpu")
            with torch.no_grad():
                encoded_frames = await asyncio.to_thread(encodec_model.encode, wav)
            codes = torch.cat([encoded[0] for encoded in encoded_frames], dim=-1).squeeze()
            codes = codes.cpu()
            semantic_tokens = semantic_tokens.cpu()
            np.savez(self.speaker_file, semantic_prompt=semantic_tokens, fine_prompt=codes, coarse_prompt=codes[:2, :])
            self.speaker_file.seek(0)

        @logger.catch()
        async def respond(self):

            await self.channel.send(content="Here is your voice file, supply it to speakgen to  use it", file=discord.File(self.speaker_file, filename=f"{self.file_name}.npz"))

            voiceclonereply_logger = logger.bind(user=self.user.name)
            voiceclonereply_logger.success("VOICECLONE Replied")

        @logger.catch()
        async def check_audio_format(self):
            self.input_audio.seek(0)
            input_data = self.input_audio.read(10)  # Read the first 10 bytes for analysis

            if input_data.startswith(b'RIFF'):
                return True
            elif input_data.startswith(b'\xFF\xFB') or input_data.startswith(b'ID3'):
                mp3_data = self.input_audio.getvalue()  # Convert MP3 to WAV using pydub
                audio_segment = AudioSegment.from_file(io.BytesIO(mp3_data), format="mp3")  # Convert mp3_data to an AudioSegment
                wav_data = audio_segment.export(format="wav").read()  # Export the AudioSegment as WAV
                self.input_audio = io.BytesIO(wav_data)  # Update self.input_audio to WAV data

                return True
            else:
                await self.channel.send(content="Must be MP3 or WAV")
                return False

        @logger.catch()
        async def check_audio_duration(self):
            self.input_audio.seek(0)
            input_data = self.input_audio.read()
            wav, sr = torchaudio.load(io.BytesIO(input_data))
            duration = wav.shape[1] / sr  # Calculate duration (number of frames divided by sample rate)
            if duration <= 30:
                return True
            else:
                await self.channel.send(content="Must be less than 30 seconds")
                return False



