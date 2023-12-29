# wordgen.py - Functions for LLM capabilities
import asyncio
import gc
import io
from io import BytesIO
import json
from loguru import logger
import discord
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration, LlamaTokenizerFast
from transformers.utils import logging as translogging
from modules.settings import SETTINGS
import warnings
from PIL import Image
import requests
import re

warnings.filterwarnings("ignore")
translogging.disable_progress_bar()
translogging.set_verbosity_error()

wordgen_user_history = {}  # This dict holds the histories for the users.


@logger.catch()
async def load_llm():
    """loads the llm"""
    if SETTINGS["usebigllm"][0] == "True":
        model_name = "llava-hf/llava-1.5-13b-hf"
        model = LlavaForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.float16, low_cpu_mem_usage=True, load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
        tokenizer = LlamaTokenizerFast.from_pretrained(model_name)
        multimodal_tokenizer = AutoProcessor.from_pretrained(model_name)
    else:
        model_name = "llava-hf/llava-1.5-7b-hf"
        model = LlavaForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.float16, low_cpu_mem_usage=True, load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
        tokenizer = LlamaTokenizerFast.from_pretrained(model_name)
        multimodal_tokenizer = AutoProcessor.from_pretrained(model_name)
    load_llm_logger = logger.bind(model=model_name)
    load_llm_logger.success("LLM Loaded.")
    return model, tokenizer, multimodal_tokenizer


async def get_defaults(idname):
    """ This function takes a filename and returns the defaults in it as a dict"""
    filename = f'defaults/{idname}.cfg'
    defaults = {}
    try:
        with open(filename, "r", encoding="utf-8") as defaults_file:
            for defaults_line in defaults_file:
                if "=" in defaults_line:
                    defaults_key, defaults_value = (defaults_line.split("=", 1)[0].strip(), defaults_line.split("=", 1)[1].strip())
                    if defaults_key in defaults:
                        if isinstance(defaults[defaults_key], list):
                            defaults[defaults_key].append(defaults_value)
                        else:
                            defaults[defaults_key] = [defaults[defaults_key], defaults_value]
                    else:
                        defaults[defaults_key] = [defaults_value]
    except FileNotFoundError:
        return None
    return defaults


class WordQueueObject:

    def __init__(self, action, metatron, user, channel, prompt=None, image_url=None, llm_prompt=None, reroll=False):
        self.action = action  # This is the queue generation action
        self.metatron = metatron  # This is the discord client
        self.user = user  # This is the discord user variable, contains user.name and user.id
        self.channel = channel  # This is the discord channel variable/
        self.model = metatron.llm_model  # This holds the current model pipeline
        self.tokenizer = metatron.llm_tokenizer
        self.multimodal_tokenizer = metatron.llm_multimodal_tokenizer
        self.prompt = prompt  # This holds the users prompt
        self.llm_prompt = llm_prompt  # This holds the llm prompt for /impersonate
        self.llm_response = None  # This holds the resulting response from generate
        self.reroll = reroll  # If this is true, when it generates text itll delete the last q/a pair and replace it with the new one.
        self.image_url = image_url

    @logger.catch()
    async def generate(self):
        """function for generating responses with the llm"""
        llm_defaults = await get_defaults('global')
        userhistory = await self.load_history()  # load the users past history to include in the prompt
        tempimage = None
        if self.reroll is True:
            await self.delete_last_history_pair()
            self.reroll = False
        if self.image_url:
            image_url = self.image_url[0]  # Consider the first image URL found
            response = requests.get(image_url)
            if response.status_code == 200:
                image_data = BytesIO(response.content)
                new_image = Image.open(image_data)
                tempimage = new_image
                image_url_pattern = r'\bhttps?://\S+\.(?:png|jpg|jpeg|gif)\S*\b'  # Updated regex pattern for image URLs
                self.prompt = re.sub(image_url_pattern, '', self.prompt)

        with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=True, enable_mem_efficient=True):  # enable flash attention for faster inference
            with torch.no_grad():
                if tempimage:
                    if self.user.id not in self.metatron.llm_user_history or not self.metatron.llm_user_history[self.user.id]:
                        formatted_prompt = f'{llm_defaults["wordsystemprompt"][0]}\n\nUSER:<image>{self.prompt}\nASSISTANT:'  # if there is no history, add the system prompt to the beginning
                    else:
                        formatted_prompt = f'{userhistory}\nUSER:<image>{self.prompt}\nASSISTANT:'
                    inputs = self.multimodal_tokenizer(formatted_prompt, tempimage, return_tensors='pt').to("cuda")
                    llm_generate_logger = logger.bind(user=self.user.name, prompt=self.prompt)
                    llm_generate_logger.info("WORDGEN Generate Started.")
                    output = await asyncio.to_thread(self.model.generate, **inputs, max_new_tokens=2000, do_sample=True)
                    llm_generate_logger.debug("WORDGEN Generate Completed")
                    result = self.multimodal_tokenizer.decode(output[0], skip_special_tokens=True)
                else:
                    if self.user.id not in self.metatron.llm_user_history or not self.metatron.llm_user_history[self.user.id]:
                        formatted_prompt = f'{llm_defaults["wordsystemprompt"][0]}\n\nUSER:{self.prompt}\nASSISTANT:'  # if there is no history, add the system prompt to the beginning
                    else:
                        formatted_prompt = f'{userhistory}\nUSER:{self.prompt}\nASSISTANT:'
                    inputs = self.tokenizer(formatted_prompt, return_tensors='pt').to("cuda")
                    llm_generate_logger = logger.bind(user=self.user.name, prompt=self.prompt)
                    llm_generate_logger.info("WORDGEN Generate Started.")
                    output = await asyncio.to_thread(self.model.generate, **inputs, max_new_tokens=2000, do_sample=True)
                    llm_generate_logger.debug("WORDGEN Generate Completed")
                    result = self.tokenizer.decode(output[0], skip_special_tokens=True)

        response_index = result.rfind("ASSISTANT:")  # this and the next line extract the bots response for posting to the channel
        self.llm_response = result[response_index + len("ASSISTANT:"):].strip()
        await self.save_history()  # save the response to the users history
        gc.collect()

    async def summary(self):
        """function for generating and posting chat summary with the llm"""
        tempimage = Image.new('RGB', (336, 336), color='black')
        channel_history = [message async for message in self.channel.history(limit=20)]
        compiled_messages = '\n'.join([f'{msg.author}: {msg.content}' for msg in channel_history])
        formatted_prompt = f'You are an AI assistant that summarizes conversations.\n\nUSER:<image> Here is the conversation to: {compiled_messages}\nASSISTANT:'  # if there is no history, add the system prompt to the beginning
        llm_summary_logger = logger.bind(user=self.user.name)
        llm_summary_logger.info("WORDGEN Summary Started.")
        with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=True, enable_mem_efficient=True):  # enable flash attention for faster inference
            with torch.no_grad():
                output = await asyncio.to_thread(self.model, tempimage, prompt=formatted_prompt, generate_kwargs={"max_length": 2048, "temperature": 0.2, "do_sample": True, "guidance_scale": 2})
        llm_summary_logger.debug("WORDGEN Summary Completed")
        response_index = output[0]["generated_text"].rfind("ASSISTANT:")  # this and the next line extract the bots response for posting to the channel
        self.llm_response = output[0]["generated_text"][response_index + len("ASSISTANT:"):].strip()
        message_chunks = [self.llm_response[i:i + 1500] for i in range(0, len(self.llm_response), 1500)]  # Post the message
        for message in message_chunks:
            await self.channel.send(message)
        llm_summary_logger = logger.bind(user=self.user.name)
        llm_summary_logger.success("SUMMARY Reply")
        gc.collect()

    async def respond(self):
        """Prints the LLM response to the chat"""
        if self.user.id in self.metatron.llm_view_last_message:  # check if there are an existing set of llm buttons for the user and if so, delete them
            try:
                await self.metatron.llm_view_last_message[self.user.id].delete()
            except discord.NotFound:
                pass

        if self.user.id in self.metatron.llm_chunks_messages:  # If its a reroll, delete the old messages
            for chunk_message in self.metatron.llm_chunks_messages[self.user.id]:
                if self.reroll:
                    try:
                        await chunk_message.delete()
                    except discord.NotFound:
                        pass  # Message not found, might have been deleted already
                    finally:
                        self.reroll = False
            del self.metatron.llm_chunks_messages[self.user.id]

        message_chunks = [self.llm_response[i:i + 1500] for i in range(0, len(self.llm_response), 1500)]  # Send and track the previously sent messages in case we have to delete them for reroll.
        if self.user.id not in self.metatron.llm_chunks_messages:
            self.metatron.llm_chunks_messages[self.user.id] = []
        for chunk in message_chunks:
            chunk_message = await self.channel.send(chunk)
            self.metatron.llm_chunks_messages[self.user.id].append(chunk_message)
        new_message = await self.channel.send(view=Wordgenbuttons(self))  # send the message with the llm buttons
        self.metatron.llm_view_last_message[self.user.id] = new_message  # track the message id of the last set of llm buttons for each user
        llm_reply_logger = logger.bind(user=self.user.name, prompt=self.prompt)
        llm_reply_logger.success("WORDGEN Reply")

    async def load_history(self):
        """loads a users history into a single string and returns it"""
        if self.user.id in self.metatron.llm_user_history and self.metatron.llm_user_history[self.user.id]:
            combined_history = ''.join(self.metatron.llm_user_history[self.user.id])
            return combined_history

    async def delete_last_history_pair(self):
        """Deletes the last question/answer pair from a users history"""
        if self.user.id in self.metatron.llm_user_history:
            self.metatron.llm_user_history[self.user.id].pop()

    async def clear_history(self):
        """deletes a users histroy"""
        if self.user.id in self.metatron.llm_user_history:
            del self.metatron.llm_user_history[self.user.id]

    async def insert_history(self):
        """inserts a question/answer pair into a users history"""
        llm_defaults = await get_defaults('global')
        if self.user.id not in self.metatron.llm_user_history:  # if they have no history, include the system prompt
            self.metatron.llm_user_history[self.user.id] = []
            injectedhistory = f'{llm_defaults["wordsystemprompt"][0]}\n\nUSER:{self.prompt}\nASSISTANT:{self.llm_prompt}</s>\n'
        else:
            injectedhistory = f'USER:{self.prompt}\nASSISTANT:{self.llm_prompt}</s>\n'
        if len(self.metatron.llm_user_history[self.user.id]) >= int(llm_defaults["wordmaxhistory"][0]):  # check if the history has reached 20 items
            del self.metatron.llm_user_history[self.user.id][0]
        self.metatron.llm_user_history[self.user.id].append(injectedhistory)

    async def save_history(self):
        """saves the prompt and llm response to the users history"""
        llm_defaults = await get_defaults('global')
        if self.user.id not in self.metatron.llm_user_history:  # if they have no history yet include the system prompt along with the special tokens for the instruction format
            self.metatron.llm_user_history[self.user.id] = []
            messagepair = f'{llm_defaults["wordsystemprompt"][0]}\n\nUSER:{self.prompt}\nASSISTANT:{self.llm_response}</s>\n'
        else:
            messagepair = f'USER:{self.prompt}\nASSISTANT:{self.llm_response}</s>\n'  # otherwise just the message pair and special tokens
        if len(self.metatron.llm_user_history[self.user.id]) >= int(llm_defaults["wordmaxhistory"][0]):  # check if the history has reached 20 items
            del self.metatron.llm_user_history[self.user.id][0]
        self.metatron.llm_user_history[self.user.id].append(messagepair)  # add the message to the history


class Wordgenbuttons(discord.ui.View):
    """Class for the ui buttons on speakgen"""

    def __init__(self, wordobject):
        super().__init__()
        self.timeout = None  # makes the buttons never time out
        self.wordobject = wordobject

    @discord.ui.button(label='Reroll last reply', emoji="🎲", style=discord.ButtonStyle.grey)
    async def reroll(self, interaction: discord.Interaction, button: discord.ui.Button):
        """Rerolls last reply"""
        if self.wordobject.user.id == interaction.user.id:
            if await self.wordobject.metatron.is_room_in_queue(self.wordobject.user.id):
                await interaction.response.send_message("Rerolling...", ephemeral=True, delete_after=5)
                self.wordobject.metatron.generation_queue_concurrency_list[interaction.user.id] += 1
                self.wordobject.reroll = True
                self.wordobject.action = "wordgen"
                await self.wordobject.metatron.generation_queue.put(self.wordobject)
            else:
                await interaction.response.send_message("Queue limit reached, please wait until your current gen or gens finish")

    @discord.ui.button(label='Delete last reply', emoji="❌", style=discord.ButtonStyle.grey)
    async def delete_message(self, interaction: discord.Interaction, button: discord.ui.Button):
        """Deletes message"""
        if self.wordobject.user.id == interaction.user.id:
            if await self.wordobject.metatron.is_room_in_queue(self.wordobject.user.id):
                self.wordobject.metatron.generation_queue_concurrency_list[interaction.user.id] += 1
                self.wordobject.action = "wordgendeletelast"
                await self.wordobject.metatron.generation_queue.put(self.wordobject)
                await interaction.response.send_message("Last question/answer pair deleted", ephemeral=True, delete_after=5)
            else:
                await interaction.response.send_message("Queue limit reached, please wait until your current gen or gens finish")

    @discord.ui.button(label='Show History', emoji="📜", style=discord.ButtonStyle.grey)
    async def dm_history(self, interaction: discord.Interaction, button: discord.ui.Button):
        """Prints history to user"""
        if self.wordobject.user.id == interaction.user.id:
            if self.wordobject.user.id in self.wordobject.metatron.llm_user_history:
                history_file = io.BytesIO(json.dumps(self.wordobject.metatron.llm_user_history[self.wordobject.user.id], indent=1).encode())
                await interaction.response.send_message(ephemeral=True, file=discord.File(history_file, filename='history.txt'))
            else:
                await interaction.response.send_message("No History", ephemeral=True, delete_after=5)
            llm_history_reply_logger = logger.bind(user=interaction.user.name)
            llm_history_reply_logger.success("WORDGEN Show History")

    @discord.ui.button(label='Wipe History', emoji="🤯", style=discord.ButtonStyle.grey)
    async def delete_history(self, interaction: discord.Interaction, button: discord.ui.Button):
        """Deletes history"""
        if self.wordobject.user.id == interaction.user.id:
            if await self.wordobject.metatron.is_room_in_queue(self.wordobject.user.id):
                self.wordobject.metatron.generation_queue_concurrency_list[interaction.user.id] += 1
                self.wordobject.action = "wordgenforget"
                await self.wordobject.metatron.generation_queue.put(self.wordobject)
                await interaction.response.send_message("History wiped", ephemeral=True, delete_after=5)
            else:
                await interaction.response.send_message("Queue limit reached, please wait until your current gen or gens finish")
