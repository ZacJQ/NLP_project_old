import cv2
import requests
import pandas as pd
from googletrans import Translator
import os
import torch
import time
from PIL import Image
from io import BytesIO
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from transformers import AutoProcessor, AutoModelForVision2Seq
from transformers.image_utils import load_image
from transformers import BitsAndBytesConfig


def download_image(image_url:str, output_folder:str)-> None:
    """
    
    """
    response = requests.get(image_url)
    if response.status_code == 200:
        filename = os.path.join(output_folder, os.path.basename(image_url))

        with open(filename, 'wb') as f:
            f.write(response.content)
        
        print(f"Image downloaded successfully: {filename}")
    else:
        print(f"Failed to download image: {image_url}")

def translate_marathi_to_english(text: str)-> str:
    """
    
    """
    translator = Translator()
    translated_text = translator.translate(text, src='mr', dest='en')
    return translated_text.text

def translate_english_to_marathi(text: str)-> str:
    """
    
    """
    translator = Translator()
    translated_text = translator.translate(text, src='en', dest='mr')
    return translated_text.text


# Example text in Marathi
marathi_text = 'नमस्ते, आपलं स्वागत आहे.'

# Translate Marathi text to English
english_translation = translate_marathi_to_english(marathi_text)
print("Marathi:", marathi_text)
print("English Translation:", english_translation)


# data_path = "Dataset/Data/Data.csv"
# data_df = pd.read_csv(data_path)

model_trans_sr_en = ""
model_vqa = "HuggingFaceM4/idefics2-8b"
model_trans_en_sr = ""
image_split = True
max_len_token_trans = 400
context_window_turns = 5
no_turns = 1


class VQA():
    def __init__(self) -> None:
        """
        Initializes the models
        """
        self.model_vqa = AutoModelForVision2Seq.from_pretrained(model_vqa,
                                                                torch_dtype=torch.float16,
                                                                quantization_config=self.quantization_config
                                                                )
        self.processor_vqa = AutoProcessor.from_pretrained(model_vqa , 
                                                           do_image_splitting=image_split
                                                           )
        self.model_trans_sr_en = AutoTokenizer.from_pretrained(model_trans_sr_en)
        self.tokenizer_sr_en = AutoModelForSeq2SeqLM.from_pretrained(model_trans_sr_en)
        self.model_language = "eng_Latn"
        global device 
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if device == 'cuda':
            self.quantization_config = BitsAndBytesConfig(
                                                    load_in_4bit=True,
                                                    bnb_4bit_quant_type="nf4",
                                                    bnb_4bit_use_double_quant=True,
                                                    bnb_4bit_compute_dtype=torch.float16
                                                    )
        else:
            self.quantization_config = BitsAndBytesConfig(
                                                    load_in_4bit=True,
                                                    bnb_4bit_quant_type="nf4",
                                                    bnb_4bit_use_double_quant=True,
                                                    bnb_4bit_compute_dtype=torch.float32
                                                    )
        pass

    def __about__(self) -> str:
        return "Class contains a instance of VQA and a translation model"
    
    def __version__(self)-> str:
        return "VQA is IDEFICS 2 and traslation model is NLLB"
    
    def __stats__(self)-> str:
        return None

    
    def get_translation(self, source_lang: str, text: str) -> str:
        """
        Converts input query into the target LLM language (English)
        """
        source_lang = "mar_Deva"  # Hard coded for now
        task = "translation"  # Hard coded for now
        translator = pipeline(task, 
                              model=self.model_trans_sr_en, 
                              tokenizer=self.tokenizer_sr_en, 
                              src_lang=source_lang, 
                              tgt_lang=self.model_language, 
                              max_length = max_len_token_trans
                              )
        output = translator(text)
        trans_text_sr_en = output[0]["translation_text"]
        self.trans_text_sr_en = trans_text_sr_en
        print(trans_text_sr_en) 
        return trans_text_sr_en

    def return_translation(self, source_lang: str)-> str:
        """
        Converts LLM output (English) to the original language 
        """
        source_lang = "mar_Deva"  # Hard coded for now
        task = "translation"  # Hard coded for now
        translator = pipeline(task, 
                              model=self.model_trans_sr_en, 
                              tokenizer=self.tokenizer_sr_en, 
                              src_lang=source_lang, 
                              tgt_lang=self.model_language, 
                              max_length = max_len_token_trans
                              )
        output = translator(self.trans_text_sr_en)
        trans_text_en_sr = output[0]["translation_text"]
        print(trans_text_en_sr) 
        return trans_text_en_sr
    
    def get_device():
        """Checks if device is available on the current device"""
        global device 
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        return device

    def quantize():
        """
        Passes quantizaton config based on the available
        """
        if device == 'cuda':
            quantization_config = BitsAndBytesConfig(
                                                    load_in_4bit=True,
                                                    bnb_4bit_quant_type="nf4",
                                                    bnb_4bit_use_double_quant=True,
                                                    bnb_4bit_compute_dtype=torch.float16
                                                    )
        else:
            quantization_config = BitsAndBytesConfig(
                                                    load_in_4bit=True,
                                                    bnb_4bit_quant_type="nf4",
                                                    bnb_4bit_use_double_quant=True,
                                                    bnb_4bit_compute_dtype=torch.float32
                                                    )
        return quantization_config
    
    def chat_template():
        return "Hey everyone"
    
    def generate_response():
        return "Yo yo yo"
    
    def append_response():
        return "Ho ho ho"
    
    def get_last_n_conversation_turns(messages: dict, no_turns: int):
        """
        Retrieve the last n conversation turns from the given messages.

        Args:
        messages (list): List of conversation messages.
        n (int): Number of conversation turns to retrieve.

        Returns:
        list: List of the last n conversation turns.
        """
        no_turns = min(no_turns, len(messages))
        conversation_turns = messages[-no_turns:]
        return conversation_turns
    
    def get_text(message: dict)-> str:
        """
        Returns the current message from the user User
        """
        user_input = message['text']
        return user_input
    
    def get_image(message)-> tuple[list,int]:
        """
        Returns the list of images and no. of images from the User
        """
        user_image = message['files']
        no_image = len(user_image)
        return (user_image, no_image)
    
    def get_template_user(self, message: str , message_history: list)-> list:
        """
        Converts the input message from user into template and appends to chat history
        """
        text = self.get_text(message)
        image_list, no_image = self.get_image(message)
        if no_image == 0:
            message_history.append({"role": "user", "content": [{"type": "text", "text": text},]})
        else:
            message_history.append({"role": "user","content": [{"type": "image"},{"type": "text", "text": text},]})
        return message_history
    
    def get_template_assistant(self, output: str, message_history: list)-> list:
        """
        Converts the LLM output into the the given template and appends to chat history
        """
        message_history.append({"role": "assistant", "content": [{"type": "text", "text": output},]})
        return message_history
    
    def chat_engine(self,messages,image: Image, max_new_token: int) -> str:
        """
        Generates assistant replies to given input
        """
        prompt = self.processor_vqa.apply_chat_template(messages, 
                                                        add_generation_prompt=True)
        if image!= None:
            inputs = self.processor_vqa(text=prompt, 
                                        images=[image], 
                                        return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            start = time.time()
            generated_ids = self.model_vqa.generate(**inputs, max_new_tokens=max_new_token)
            print("time for generations:", (time.time() - start))
            print("max memory allocated:", (torch.cuda.max_memory_allocated())/1024*1024)
            print("number of tokens generated:", len(generated_ids[:, 
                                                                inputs["input_ids"].size(1):][0]
                                                                    ))
            output = self.processor_vqa.batch_decode(generated_ids, skip_special_tokens=True)
            print(self.processor_vqa.batch_decode(generated_ids, skip_special_tokens=True))
            return output
        else:
            inputs = self.processor_vqa(text=prompt, 
                                        return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            start = time.time()
            generated_ids = self.model_vqa.generate(**inputs, max_new_tokens=max_new_token)
            print("time for generations:", (time.time() - start))
            print("max memory allocated:", (torch.cuda.max_memory_allocated())/1024*1024)
            print("number of tokens generated:", len(generated_ids[:, 
                                                                inputs["input_ids"].size(1):][0]
                                                                    ))
            output = self.processor_vqa.batch_decode(generated_ids, skip_special_tokens=True)
            print(self.processor_vqa.batch_decode(generated_ids, skip_special_tokens=True))
            return output

    
global chat_history
chat_history = []
vqa = VQA()

def chat_vqa(messages: dict,history: dict):
    input = vqa.get_template_user(message=messages, message_history=chat_history)
    image, _ = vqa.get_image(messages)
    output = vqa.chat_engine(messages=input, image=image, max_new_token=512)
    chat_history = vqa.get_template_assistant(output=output,message_history=input)
    
