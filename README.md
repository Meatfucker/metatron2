# metatron2
A rewrite of metatron without dependencies on APIs with machine learning functions, including LLM chat, Image generation, and Speech Generation capabilities

LLM implemented by chatting with it, simply tag the bot with a message. there is also an /impersonate command for few-shot prompting.
Audio generation is implemented via /speakgen. By default it uses the base bark voice but voice files are supported. See https://github.com/C0untFloyd/bark-gui for details on how to make custom voices
Image generation has rudimentary support via /imagegen. Currently supports A1111 style prompt weights, model loading, embedding loading, and batch size. Rest of the expected features are in the works.
settings.cfg has changed a bit from metatron, see the example file.
