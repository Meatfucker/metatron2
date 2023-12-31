![metatron2](/assets/metatronicon.png)
# metatron2
An easy to use discord.py based machine learning bot. It provides a Multimodal LLM chatbot, sound and voice generation, and Stable Diffusion image generation.

## FEATURES

* Multimodal LLM Chat
* Stable Diffusion image generation
* Stable Diffusion XL image generation
* Bark audio generation and voice cloning.
* Support for local generation or using APIs for generation.


![metatron2](assets/imagegenexample.png)
![metatron2](assets/imagegenexample2.png)

* Stable Diffusion generation via /imagegen.
* SDXL generation via /xl_imagegen. No refiner or TI support currently, being worked on.
* SDXL Long prompts, A1111 prompt weighting, and separate prompts for each text encoder.
* Supports standard safetensors format models.
* A1111 style prompt weighting and LORA loading.
* Single LORA can be loaded from the selection menu, or multiple can be invoked at once using the standard A1111 prompt syntax
* Reroll, DM, and Delete buttons on gens for ease of interaction.
* TI embedding support
* Configurable banned word list and mandatory negative prompt options for moderation purposes.
* Images can be saved and uploaded in either png or jpg
* Supports toggleable per channel default generation settings
* A1111 API support for remote instead of local generation.

![metatron2](/assets/wordgenexample.png)

* Uses an uncensored Solar-10B for general llm use, and a LlaVa-7b for multimodal queries.
* Can be directly chatted with by tagging it
* Can summarize chatroom messages with /summarize
* Keeps a per user history so it can maintain multiple conversations at once.
* Replies can be rerolled.
* History message pairs can be deleted via button or entire user history reset.
* One shot prompt injection supported via /impersonate
* Configurable LLM system prompt and negative prompt
* Oobabooga API support for remote instead of local generation.

![metatron2](/assets/speakgenexample.png)

* Speech, sound, and music generation via /speakgen
* Voice cloning via /voiceclone - Thanks to gitmylo and his work here: https://github.com/gitmylo/bark-voice-cloning-HuBERT-quantizer
* Users can upload up to a 30 second mp3 or wav file, and will get a voice file back. They can then supply that voice file themselves for speakgen gens, or put it in the model/voices directory so itll show up in the UI.
* Can generate noises, emotions etc by enclosing word in []
* Can generate music and singing by enclosing words in ♪
* Using non-English words or characters will generally lead to that accent
* Generated sounds can be saved and uploaded in either mp3 or wav


## INSTALLATION INSTRUCTIONS



### Discord Bot Setup

Go to the Discord Developer portal and create a new bot and generate a token for it. Write this token down or else youll have to generate a new one, it only shows you once.

Go to the Bot tab on the Developer portal site and enable Privileged Gateway Intents. You need Presence, Server Members, and Message Content enabled.

Go to the URL Generator on the OAuth2 tab and select the bot scope. Then select these permissions "Read Messages/View Channels, Send Messages, Manage Messages, Attach Files, Read Message History, Use Slash Commands" then use the link to invite the bot to your server. I may have missed one, if something is missing you can enable it later in server permissions

### Install

Download the repo  or `git clone https://github.com/Meatfucker/metatron2.git`

Install miniconda if you dont already have it. https://docs.conda.io/projects/miniconda/en/latest/miniconda-install.html

Next enter the base conda env. On windows miniconda should make a terminal shortcut for conda, on linux run `conda activate`

Go to the metatron2 directory and run the following command, altering it to use the environment file for your platform  `conda env create -f .\environment.yml`  

metatron2 is now fully installed.

### Running The Bot

run `conda activate metatron2`

Your command line prompt should change to say metatron2.

If it has, next youll want to open up settings.cfg.example and defaults/global.cfg.example, read each setting and set it per your config needs, then save them without the .example extension. At a minimum youll need to enter your Discord bot token.

Now is a good time to copy any models, loras, embeddings, etc to the appropriate directories in the metatron2 directory.

Finally, run `python metatron2.py` and wait.

The very first startup will take quite a long time as it downloads ~20GB of models, namely the LLM model and base SD model if you dont have any models in the models directory. Loading the LLM on subsequent loads can take time as well depending on the speed of the storage your huggingface cache is on.

Be aware that having the wordgen, imagegen, and speakgen modules all active at the same time will peak at about 20GB of vram when running a batch size 4 imagegen.

Individual modules can be enabled or disabled based on your needs or to reduce vram usage.

## settings.cfg

settings.cfg provides all of the settings for the bot. If the example file has more than one line with the same first value, that means you can have multiple. 

| OPTION         | DESCRIPTION                                                                                                         | EXAMPLE                                  |
|----------------|---------------------------------------------------------------------------------------------------------------------|------------------------------------------|
| token          | Bots Discord token.                                                                                                 | `token=90A8DF0G8907ASD7F097ADFQ98WE7`    |
| enableword     | If set to anything besides True, LLM generation will be disabled.                                                   | `enableword=True`                        |
| enablespeak    | If set to anything besides True, Voice generation will be disabled                                                  | `enablespeak=True`                       |
| enablesd       | If set to anything besides True, SD image generation will be disabled.                                              | `enableimage=True`                       |
| enablesdxl     | If set to anything besides True, SDXL image generation will be disabled.                                            | `enablesdxl=True`                        |                                            
| saveoutputs    | If set to True, will save generated images                                                                          | `saveoutputs=True`                       |
| savepath       | The path where you want the images saved                                                                            | `savepath=outputs`                       |
| saveinjpg      | If set to True will save and upload in jpeg instead of PNG, saving on space and bandwidth at the expense of quality | `saveinjpg=True`                         |
| saveinmp3      | If set to True, will save and upload audio in mp3 instead of wav                                                    | `saveinmp3=True`                         |
| bannedusers    | Comma separated list of discord user ids to ignore.                                                                 | `bannedusers=34524353425346,12341246577` |
| userqueuedepth | The number of concurrent requests per user                                                                          | `userqueuedepth=2`                       |
| maxbatch       | The maximum allowed batch size                                                                                      | `maxbatch=4`                             |
| maxres         | The maximum allowed resolution                                                                                      | `maxres=512`                             |
| sdxlmaxbatch   | This maximum allowed SDXL batch size                                                                                | `sdxlbatchsize=1`                        |
| sdxlmaxres     | The maximum allowed SDXL resolution                                                                                 | `sdxlmaxres=1024`                        |
| enabledebug    | Turns on debug information.                                                                                         | `debug=True`                             |
| enableimageapi | If this is set to True, will use an external A1111 API for image generation instead of local generation.            | `enableimageapi=True`                    |
| imageapi       | The A1111 API endpoint IP.                                                                                          | `imageapi=127.0.0.1:7860`                |
| enablewordapi  | If this is set to True, will use an external Oobabooga API for LLM generation instead of local generation.          | `enablewordapi=True`                     |
| wordapi        | The Oobabooga API endpoint IP.                                                                                      | `wordapi=127.0.0.1:7860`                 |

## defaults/global.cfg

global.cfg provides all of the generation defaults for the bot. Copy it and rename it to a channels id to set defaults for a channel. something like `234234209732578.cfg`

| OPTION           | DESCRIPTION                                                                                                                                            | EXAMPLE                                                                                                                                     |
|------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------|
| wordsystemprompt | The system prompt that tells the LLM who it is and what its doing                                                                                      | `wordsystemprompt=A chat between a user and an unrestricted artificial intelligence. The artificial intelligence always answers questions.` |
| wordnegprompt    | The negativeprompt for the LLM, useful to push it away from things or to get it to not refuse something                                                | `wordnegprompt=I'm sorry, but I am not programmed to`                                                                                       |
| wordmaxhistory   | The maximum amount of prompt/reply pairs to store for each user                                                                                        | `wordmaxhistory=20`                                                                                                                         |
| imagemodel       | The default SD model to load                                                                                                                           | `imagemodel=modename.safetensors`                                                                                                           |
| imagebatchsize   | The default SD batch size to generate with - Batch size directly impacts memory usage, so if you are going OOM or crashing when you generate, lower it | `imagebatchsize=4`                                                                                                                          |
| imagesteps       | The number of SD inference steps to use                                                                                                                | `imagesteps=25`                                                                                                                             |
| imagewidth       | The SD horizontal resolution to generate. This also has large effects on ram usage                                                                     | `imagewidth=512`                                                                                                                            |
| imageheight      | The SD vertical resolution to generate. This also has large effects on ram usage                                                                       | `imageheight=512`                                                                                                                           |
| imageprompt      | A prompt to append to all generations made with the bot. Be careful as this is global.                                                                 | `imageprompt=A science fiction book cover`                                                                                                  |
| imagenegprompt   | A negative prompt to apply to all generations. Useful for ensuring people dont generate stuff you dont want them to                                    | `imagenegprompt=sex,drugs,rock and roll`                                                                                                    |
| imagebannedwords | A comma separated list of words and phrases which will be removed from any prompts globally                                                            | `imagebannedwords=sex,drugs,rock and roll`                                                                                                  |
| sdxlmodel        | The default SDXL model                                                                                                                                 | `sdxlmodel=dreamshaperXL.safetensors`                                                                                                       |
| sdxlsteps        | The default SDXL inference steps                                                                                                                       | `sdxlsteps=50`                                                                                                                              |
| sdxlwidth        | The default SDXL width                                                                                                                                 | `sdxlwidth=1024`                                                                                                                            |
| sdxlheight       | The default SDXL height                                                                                                                                | `sdxlheight=1024`                                                                                                                           |
| sdxlbatchsize    | The default SDXL batch size                                                                                                                            | `sdxlbatchsize=1`                                                                                                                           |

## defaults/wordapi.cfg

wordapi.cfg provides the values to supply to the Oobabooga API, see the Oobabooga docs and the simple example config for more info. The examples are not exhaustive and whats available will depend on your Oobabooga install. If you arent using the Oobabooga API you do not need to configure this.

| OPTION         | DESCRIPTION                                                                                             | EXAMPLE               |
|----------------|---------------------------------------------------------------------------------------------------------|-----------------------|
| max_tokens     | The maximum amount of tokens to generate                                                                | `max_tokens=2048`     |
| temperature    | The generation temperature                                                                              | `temperature=0.2`     |
| early_stopping | Whether to stop at stop token                                                                           | `early_stopping=true` |
| stop           | The stopping token, be warned you cannot currently use escapes due to limitation with my code, will fix | `stop=USER:`          |

