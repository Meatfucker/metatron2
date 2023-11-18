#wordgen.py - Functions for LLM capabilities
import os
os.environ["TQDM_DISABLE"] = "1" #Attempt to turn off the annoying progress bar
import asyncio
import io
import json
import sys
import time
from loguru import logger
import discord
import torch
from transformers import LlamaForCausalLM, LlamaConfig, LlamaTokenizer
from transformers.utils import logging as translogging
from modules.settings import SETTINGS, get_defaults

translogging.set_verbosity_error() #Try to silence transformers logging spam
logger.remove() #More of the same
wordgen_user_history = {} #This dict holds the histories for the users.

@logger.catch
async def load_llm():
    '''loads the llm'''
    model = LlamaForCausalLM.from_pretrained("liuhaotian/llava-v1.5-13b", load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True, low_cpu_mem_usage=True, device_map="auto") #load model
    model = model.to_bettertransformer() #Use bettertransformers for more speed
    model.eval()
    logger.success("LLM Model Loaded.")
    tokenizer = LlamaTokenizer.from_pretrained("liuhaotian/llava-v1.5-13b") #load tokenizer
    logger.success("LLM Tokenizer Loaded.")
    return model, tokenizer

@logger.catch    
async def llm_generate(message, prompt, model, tokenizer):
    '''function for generating responses with the llm'''
    llm_defaults = await get_defaults('global')
    userhistory = await load_history(message) #load the users past history to include in the prompt
    if message.author.id not in wordgen_user_history or not wordgen_user_history[message.author.id]:
        formatted_prompt = f'{llm_defaults["wordsystemprompt"][0]}\n\nUSER:{prompt}\nASSISTANT:' #if there is no history, add the system prompt to the beginning
    else:
        formatted_prompt = f'{userhistory}\nUSER:{prompt}\nASSISTANT:'
    input_ids = tokenizer.encode(formatted_prompt, return_tensors="pt") #turn prompt into tokens
    input_ids = input_ids.to('cuda') #send tokens to gpu
    negative_input_ids = tokenizer.encode(llm_defaults["wordnegprompt"][0], return_tensors="pt") #turn negative prompt into tokens
    negative_input_ids = negative_input_ids.to('cuda') #negative tokens to gpu
    llm_generate_logger = logger.bind(user=message.author.name, userid=message.author.id, prompt=prompt)
    llm_generate_logger.debug("WORDGEN Generate Started.")
    with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False): #enable flash attention for faster inference
        output = await asyncio.to_thread(model.generate, input_ids, max_length=4096, temperature=0.2, do_sample=True, guidance_scale=2, negative_prompt_ids=negative_input_ids) #run the inference in a thread so it doesnt block the bots execution
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True) #turn the returned tokens into string
    llm_generate_logger.debug("WORDGEN Generate Completed")
    response_index = generated_text.rfind("ASSISTANT:") #this and the next line extract the bots response for posting to the channel
    llm_response = generated_text[response_index + len("ASSISTANT:"):].strip()
    await save_history(generated_text, message) #save the response to the users history
    return(llm_response)
 
@logger.catch 
async def save_history(generated_text, message):
    '''saves the prompt and llm response to the users history'''
    llm_defaults = await get_defaults('global') #get default values
    last_message_index = generated_text.rfind("USER:") #this and the next line extract the last question/answer pair from the generated text
    last_message_pair = generated_text[last_message_index:].strip()
    if message.author.id not in wordgen_user_history: #if they have no history yet include the system prompt along with the special tokens for the instruction format
        wordgen_user_history[message.author.id] = []
        messagepair = f'{llm_defaults["wordsystemprompt"][0]}\n\n{last_message_pair}</s>\n'
    else:
        messagepair = f'{last_message_pair}</s>\n' #otherwise just the message pair and special tokens
    if len(wordgen_user_history[message.author.id]) >= int(llm_defaults["wordmaxhistory"][0]):  # check if the history has reached 20 items
        del wordgen_user_history[message.author.id][0]
    wordgen_user_history[message.author.id].append(messagepair) #add the message to the history

@logger.catch    
async def load_history(message):
    '''loads a users history into a single string and returns it'''
    if message.author.id in wordgen_user_history and wordgen_user_history[message.author.id]:
        combined_history = ''.join(wordgen_user_history[message.author.id])
        return combined_history

@logger.catch    
async def clear_history(message):
    '''deletes a users history'''
    if message.author.id in wordgen_user_history:
        del wordgen_user_history[message.author.id]

@logger.catch
async def delete_last_history(message):
    '''deletes the last question/answer pair from a users history'''
    if message.author.id in wordgen_user_history:
        wordgen_user_history[message.author.id].pop()

@logger.catch
async def insert_history(userid, prompt, llm_prompt):
    '''inserts a question/answer pair into a users history'''
    llm_defaults = await get_defaults('global')
    if userid not in wordgen_user_history: #if they have no history, include the system prompt
        wordgen_user_history[userid] = []
        injectedhistory = f'{llm_defaults["wordsystemprompt"][0]}\n\nUSER:{prompt}\nASSISTANT:{llm_prompt}</s>\n'
    else:
        injectedhistory = f'USER:{prompt}\nASSISTANT:{llm_prompt}</s>\n'
    wordgen_user_history[userid].append(injectedhistory)
        
class Wordgenbuttons(discord.ui.View):
    '''Class for the ui buttons on speakgen'''

    def __init__(self, generation_queue, userid, message, stripped_message, metatron_client):
        super().__init__()
        self.timeout = None #makes the buttons never time out
        self.generation_queue = generation_queue
        self.userid = userid
        self.message = message
        self.stripped_message = stripped_message
        self.metatron_client = metatron_client

    @logger.catch
    @discord.ui.button(label='Reroll last reply', emoji="🎲", style=discord.ButtonStyle.grey)
    async def reroll(self, interaction: discord.Interaction, button: discord.ui.Button):
        """Rerolls last reply"""
        if self.userid == interaction.user.id:
            if await self.metatron_client.is_room_in_queue(self.userid) == True:
                await interaction.response.send_message("Rerolling...", ephemeral=True, delete_after=5)
                #await self.generation_queue.put(('wordgendeletelast', self.userid, self.message, self.stripped_message))
                await self.generation_queue.put(('wordgengenerate', self.userid, self.message, self.stripped_message, True))
            else:
                await interaction.response.send_message("Queue limit reached, please wait until your current gen or gens finish")

    @logger.catch
    @discord.ui.button(label='Delete last reply', emoji="❌", style=discord.ButtonStyle.grey)
    async def delete_message(self, interaction: discord.Interaction, button: discord.ui.Button):
        """Deletes message"""
        if self.userid == interaction.user.id:
            if await self.metatron_client.is_room_in_queue(self.userid) == True:
                await self.generation_queue.put(('wordgendeletelast', self.userid, self.message, self.stripped_message))
                await interaction.response.send_message("Last question/answer pair deleted", ephemeral=True, delete_after=5)
            else:
                await interaction.response.send_message("Queue limit reached, please wait until your current gen or gens finish")

    @logger.catch
    @discord.ui.button(label='Show History', emoji="📜", style=discord.ButtonStyle.grey)
    async def dm_history(self, interaction: discord.Interaction, button: discord.ui.Button):
        """Prints history to user"""
        if self.userid == interaction.user.id:
            if self.message.author.id in wordgen_user_history:
                history_file = io.BytesIO(json.dumps(wordgen_user_history[self.userid], indent=1).encode()) #turn the users history into a txt file-like object
                await interaction.response.send_message(ephemeral=True, file=discord.File(history_file, filename='history.txt'))
            else:
                await interaction.response.send_message("No History", ephemeral=True, delete_after=5)
            self.llm_history_reply_logger = logger.bind(user=interaction.user.name, userid=interaction.user.id)
            self.llm_history_reply_logger.success("WORDGEN Show History")
    
    @logger.catch
    @discord.ui.button(label='Wipe History', emoji="🤯", style=discord.ButtonStyle.grey)
    async def delete_history(self, interaction: discord.Interaction, button: discord.ui.Button):
        """Deletes history"""
        if self.userid == interaction.user.id:
            if await self.metatron_client.is_room_in_queue(self.userid) == True:
                await self.generation_queue.put(('wordgenforget', self.userid, self.message, self.stripped_message))
                await interaction.response.send_message("History wiped", ephemeral=True, delete_after=5)
            else:
                await interaction.response.send_message("Queue limit reached, please wait until your current gen or gens finish")