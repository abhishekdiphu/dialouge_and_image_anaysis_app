import os

from transformers import  AutoTokenizer, AutoModelForSeq2SeqLM, GenerationConfig
from peft import PeftModel, PeftConfig
import torch
import numpy as np
import pandas as pd

# tqdm library makes the loops show a smart progress meter.
from tqdm import tqdm
tqdm.pandas()

from flask import jsonify


def summerize_text(text):
    instruction = "Summarize the following conversation. "
    prompt = instruction + text.strip()
    print("Prompt: #########################", prompt)
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("Device:##############################", device)
    if device == "mps":
        torch.set_default_dtype(torch.float32)
    
    #base_model 
    model_name="google/flan-t5-base"
    peft_path = "./src/saved_models/ppo_model"  # contains adapter_config.json + adapter weights
    cfg = PeftConfig.from_pretrained(peft_path)

    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model_name_or_path)
    base = AutoModelForSeq2SeqLM.from_pretrained(cfg.base_model_name_or_path)
    ppo_model = PeftModel.from_pretrained(base, peft_path).to(device)
    tokenizer = AutoTokenizer.from_pretrained("src/saved_models/ppo_model", device_map="auto")
    
    input_ids= tokenizer( prompt, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
    
    
    max_new_tokens=100
    generation_config = GenerationConfig(max_new_tokens=max_new_tokens,
                                        top_k=50,
                                        top_p=1.0,
                                        do_sample=True)
    with torch.no_grad():
        response_token_ids = ppo_model.generate(**input_ids,
                                            generation_config=generation_config)
        
    
    generated_text = tokenizer.decode(response_token_ids[0], skip_special_tokens=True)
    print("Generated Text: #########################", generated_text)

    return jsonify({"summary": generated_text})


   
    
