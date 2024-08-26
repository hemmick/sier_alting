#!/usr/bin/env python3
import os,sys
from TTS.api import TTS
from transformers import NllbTokenizer, AutoModelForSeq2SeqLM

def to_engelsk(norsk):
    modstr="facebook/nllb-200-distilled-600M"

    tokenizer = NllbTokenizer.from_pretrained(modstr)
    model = AutoModelForSeq2SeqLM.from_pretrained(modstr)
    norsk = f"<2nob_Latn> {norsk}"

    inputs = tokenizer(norsk, return_tensors="pt")
    engelsk_tokens = model.generate(**inputs, forced_bos_token_id=tokenizer.convert_tokens_to_ids('2eng_Latn'))
    
    print(repr(engelsk_tokens))
    
    engelsk = tokenizer.batch_decode(engelsk_tokens, skip_special_tokens=True)
    print(repr(engelsk))

    print(tokenizer(norsk).input_ids)
    print(" ".join(engelsk) + "\n")
    return engelsk

def collect_norwegian_text():
    norwegian_text = ""
    print("Enter your Norwegian text (press Enter on a new line when done):")
    while True:
        line = input()
        if line == "":
            break
        norwegian_text += line + "\n"
    return norwegian_text

def norsk_speak(norsk_text):
    # Initialize TTS
    model = "tts_models/multilingual/multi-dataset/bark"
    tts = TTS(model)

    # Synthesize speech and save to a file
    tts.tts_to_file(text=text, file_path="/tmp/norsk_audio.wav")
    os.system(f'afplay /tmp/norsk_audio.wav')

# Check if there's a command-line argument
if len(sys.argv) > 1:
    norwegian_text = " ".join(sys.argv[1:])
else:
    # Collect the Norwegian text from the user
    norwegian_text = collect_norwegian_text()
print("\n")
engelsk = to_engelsk(norwegian_text)
print(repr(engelsk))

norsk_speak(norwegian_text)

#play audio
norsk_speak()

