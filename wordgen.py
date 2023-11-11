#wordgen.py - Functions for LLM capabilities

from transformers import LlamaForCausalLM, LlamaConfig, LlamaTokenizer
import os
import sys
import time
import torch
import asyncio

wordgen_user_history = {}

async def load_llm():

    model = LlamaForCausalLM.from_pretrained("liuhaotian/llava-v1.5-13b", load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True, low_cpu_mem_usage=True, device_map="auto")
    model.eval()
    tokenizer = LlamaTokenizer.from_pretrained("liuhaotian/llava-v1.5-13b")
    return model, tokenizer
    
async def wordgen(message, prompt, model, tokenizer, systemprompt="You are an AI", negativeprompt=""):
    
    userhistory = await loadhistory(message)
    print(f'USER HISTORY {userhistory}')
    if message.author.id not in wordgen_user_history or not wordgen_user_history[message.author.id]:
        formattedprompt = f'{systemprompt}\n\n USER: {prompt}\nASSISTANT:'
    else:
        formattedprompt = f'{userhistory}\n USER: {prompt}\nASSISTANT:'
    input_ids = tokenizer.encode(formattedprompt, return_tensors="pt")
    input_ids = input_ids.to('cuda')
    negative_input_ids = tokenizer.encode(negativeprompt, return_tensors="pt")
    negative_input_ids = negative_input_ids.to('cuda')
    output = await asyncio.to_thread(model.generate, input_ids, max_length=4096, temperature=0.2, do_sample=True, guidance_scale=2, negative_prompt_ids=negative_input_ids)
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    response_index = generated_text.rfind("ASSISTANT:")
    llmresponse = generated_text[response_index + len("ASSISTANT:"):].strip()
    await savehistory(generated_text, message, systemprompt)
    print(wordgen_user_history)
    return(llmresponse)
    
async def savehistory(generated_text, message, systemprompt):

    last_message_index = generated_text.rfind("USER:")
    last_message_pair = generated_text[last_message_index:].strip()
    if message.author.id not in wordgen_user_history:
        wordgen_user_history[message.author.id] = []
        messagepair = f'{systemprompt}\n\n{last_message_pair}</s>\n'
    else:
        messagepair = f'{last_message_pair}</s>\n'
    wordgen_user_history[message.author.id].append(messagepair)
    
async def loadhistory(message):
    
    if message.author.id in wordgen_user_history and wordgen_user_history[message.author.id]:
        # Combine all strings in the user's history list into one string
        combined_history = ''.join(wordgen_user_history[message.author.id])
        return combined_history
    else:
        return None

async def clearhistory(message):
    if message.author.id in wordgen_user_history:
        del wordgen_user_history[message.author.id]