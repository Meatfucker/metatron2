#wordgen.py - Functions for LLM capabilities

from transformers import LlamaForCausalLM, LlamaConfig, LlamaTokenizer
import os
import sys
import time
import torch
import io
import json
import asyncio
from loguru import logger
import discord

wordgen_user_history = {}

async def load_llm():
    '''loads the llm'''
    model = LlamaForCausalLM.from_pretrained("liuhaotian/llava-v1.5-13b", load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True, low_cpu_mem_usage=True, device_map="auto") #load model
    model = model.to_bettertransformer() #Use bettertransformers for more speed
    model.eval()
    logger.success("Model Loaded.")
    tokenizer = LlamaTokenizer.from_pretrained("liuhaotian/llava-v1.5-13b") #load tokenizer
    logger.success("Tokenizer Loaded.")
    return model, tokenizer
    
async def wordgen(message, prompt, model, tokenizer, systemprompt="You are an AI", negativeprompt=""):
    '''function for generating responses with the llm'''
    userhistory = await loadhistory(message) #load the users past history to include in the prompt
    if message.author.id not in wordgen_user_history or not wordgen_user_history[message.author.id]:
        formattedprompt = f'{systemprompt}\n\nUSER:{prompt}\nASSISTANT:' #if there is no history, add the system prompt to the beginning
    else:
        formattedprompt = f'{userhistory}\nUSER:{prompt}\nASSISTANT:'
    input_ids = tokenizer.encode(formattedprompt, return_tensors="pt") #turn prompt into tokens
    input_ids = input_ids.to('cuda') #send tokens to gpu
    negative_input_ids = tokenizer.encode(negativeprompt, return_tensors="pt") #turn negative prompt into tokens
    negative_input_ids = negative_input_ids.to('cuda') #negative tokens to gpu
    logger.info("WORDGEN Generate Started.")
    with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False): #enable flash attention for faster inference
        output = await asyncio.to_thread(model.generate, input_ids, max_length=4096, temperature=0.2, do_sample=True, guidance_scale=2, negative_prompt_ids=negative_input_ids) #run the inference in a thread so it doesnt block the bots execution
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True) #turn the returned tokens into string
    logger.success("WORDGEN Generate Completed")
    response_index = generated_text.rfind("ASSISTANT:") #this and the next line extract the bots response for posting to the channel
    llmresponse = generated_text[response_index + len("ASSISTANT:"):].strip()
    await savehistory(generated_text, message, systemprompt) #save the response to the users history
    return(llmresponse)
    
async def savehistory(generated_text, message, systemprompt):
    '''saves the prompt and llm response to the users history'''
    last_message_index = generated_text.rfind("USER:") #this and the next line extract the last question/answer pair from the generated text
    last_message_pair = generated_text[last_message_index:].strip()
    if message.author.id not in wordgen_user_history: #if they have no history yet include the system prompt along with the special tokens for the instruction format
        wordgen_user_history[message.author.id] = []
        messagepair = f'{systemprompt}\n\n{last_message_pair}</s>\n'
    else:
        messagepair = f'{last_message_pair}</s>\n'
    wordgen_user_history[message.author.id].append(messagepair) #add the message to the history
    
async def loadhistory(message):
    
    if message.author.id in wordgen_user_history and wordgen_user_history[message.author.id]:
        combined_history = ''.join(wordgen_user_history[message.author.id])
        return combined_history
    
async def clearhistory(message):
    
    if message.author.id in wordgen_user_history:
        del wordgen_user_history[message.author.id]

async def deletelasthistory(message):
    
    if message.author.id in wordgen_user_history:
        wordgen_user_history[message.author.id].pop()

async def inserthistory(userid, prompt, llmprompt, systemprompt):

    if userid not in wordgen_user_history:
        wordgen_user_history[userid] = []
        injectedhistory = f'{systemprompt}\n\nUSER:{prompt}\nASSISTANT:{llmprompt}</s>\n'
    else:
        injectedhistory = f'USER:{prompt}\nASSISTANT:{llmprompt}</s>\n'
    wordgen_user_history[userid].append(injectedhistory)
        
class Wordgenbuttons(discord.ui.View):
    """Class for the ui buttons on speakgen"""

    def __init__(self, wordgen_queue, userid, message, strippedmessage):
        super().__init__()
        self.timeout = None
        self.wordgen_queue = wordgen_queue
        self.userid = userid
        self.message = message
        self.strippedmessage = strippedmessage

    @discord.ui.button(label='Reroll last reply', emoji="🎲", style=discord.ButtonStyle.grey)
    @logger.catch   
    async def reroll(self, interaction: discord.Interaction, button: discord.ui.Button):
        """Rerolls last reply"""
        if self.userid == interaction.user.id:
            await interaction.response.send_message("Rerolling...", ephemeral=True, delete_after=5)
            await self.wordgen_queue.put(('wordgendeletelast', self.message, self.strippedmessage))
            await self.wordgen_queue.put(('wordgengenerate', self.message, self.strippedmessage))
            

    @discord.ui.button(label='Delete last reply', emoji="❌", style=discord.ButtonStyle.grey)
    @logger.catch
    async def delete_message(self, interaction: discord.Interaction, button: discord.ui.Button):
        """Deletes message"""
        if self.userid == interaction.user.id:
            await self.wordgen_queue.put(('wordgendeletelast', self.message, self.strippedmessage))
            await interaction.response.send_message("Last question/answer pair deleted", ephemeral=True, delete_after=5)
            
         
    @discord.ui.button(label='Show History', emoji="📜", style=discord.ButtonStyle.grey)
    @logger.catch   
    async def dm_history(self, interaction: discord.Interaction, button: discord.ui.Button):
        """Prints history to user"""
        if self.userid == interaction.user.id:
            if self.message.author.id in wordgen_user_history:
                historyfile = io.BytesIO(json.dumps(wordgen_user_history[self.userid], indent=1).encode())
                await interaction.response.send_message(ephemeral=True, file=discord.File(historyfile, filename='history.txt'))
            else:
                await interaction.response.send_message("No History", ephemeral=True, delete_after=5)
            logger.info("WORDGEN Show History.")
         
    @discord.ui.button(label='Wipe History', emoji="🤯", style=discord.ButtonStyle.grey)
    @logger.catch   
    async def delete_history(self, interaction: discord.Interaction, button: discord.ui.Button):
        """Deletes history"""
        if self.userid == interaction.user.id:
            await self.wordgen_queue.put(('wordgenforget', self.message, self.strippedmessage))
            await interaction.response.send_message("History wiped", ephemeral=True, delete_after=5)



