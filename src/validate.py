import os
os.environ["TRANSFORMERS_NO_TF"] = "1"  # PyTorch-only


from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM, GenerationConfig
from datasets import load_dataset
from peft import PeftModel, PeftConfig, LoraConfig, TaskType

# trl: Transformer Reinforcement Learning library
from trl import PPOTrainer, PPOConfig, AutoModelForSeq2SeqLMWithValueHead
from trl import create_reference_model
from trl.core import LengthSampler

import torch
import evaluate

import numpy as np
import pandas as pd

# tqdm library makes the loops show a smart progress meter.
from tqdm import tqdm
tqdm.pandas()

from paths import PATH_PARAMS
from utility import RlhfUtils as rl
from utility import EvaluationTools as Eval



print_debug = True
#device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("Device:##############################", device)
    if device == "mps":
        torch.set_default_dtype(torch.float32)
    #device = torch.device("cpu")

    env_configs_file = PATH_PARAMS["configs"]
    env_configs = rl.load_configs(env_configs_file)
    print_debug = True

    #base_model 
    model_name="google/flan-t5-base"
    
    #dataset
    huggingface_dataset_name = "knkarthick/dialogsum"
    dataset_original = rl.load_data(huggingface_dataset_name)
    #dataset_original.save_to_disk("dataset_rlhf")
    if print_debug:
        print(dataset_original)
    
    #dataset alighned for the base model
    dataset = rl.build_dataset(model_name=model_name,
                        dataset_name=huggingface_dataset_name,
                        input_min_text_length=200, 
                        input_max_text_length=1000)
    print("Dataset after processing:", dataset)    

    #prepare the PPO model 
   

    peft_path = "./saved_models/ppo_model"  # contains adapter_config.json + adapter weights
    cfg = PeftConfig.from_pretrained(peft_path)

    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model_name_or_path)
    base = AutoModelForSeq2SeqLM.from_pretrained(cfg.base_model_name_or_path)
    ppo_model = PeftModel.from_pretrained(base, peft_path).to(device)
   
    print(f'PPO model parameters to be updated (ValueHead + 769 params):\n{rl.print_number_of_trainable_model_parameters(ppo_model)}\n')
    
    # Reference model
    ref_lora_config = LoraConfig(
    r=32, # Rank
    lora_alpha=32,
    target_modules=["q", "v"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.SEQ_2_SEQ_LM # FLAN-T5
    )
    ref_model = AutoModelForSeq2SeqLM.from_pretrained(model_name, 
                                                torch_dtype=torch.float32)
    ref_peft_model = PeftModel.from_pretrained(ref_model, 
                                        './peft-dialogue-summary-checkpoint-from-s3/', 
                                        lora_config=ref_lora_config,
                                        torch_dtype=torch.float32, 
                                        device_map="auto",                                       
                                        is_trainable=False)


    ref_ppo_model = AutoModelForSeq2SeqLMWithValueHead.from_pretrained(ref_peft_model,                                                               
                                                               torch_dtype=torch.float32,
                                                               is_trainable=False).to(device)
    ref_model = create_reference_model(ref_ppo_model)
    print(f'ref model parameters to be updated (ValueHead + 769 params):\n{rl.print_number_of_trainable_model_parameters(ref_model)}\n')



    # prepare reqward model - toxicity model
    toxicity_model_name = "facebook/roberta-hate-speech-dynabench-r4-target"
    
    
    
    toxicity_tokenizer = AutoTokenizer.from_pretrained("saved_models/toxicity-reward-model")
    toxicity_model = AutoModelForSequenceClassification.from_pretrained(
                    "saved_models/toxicity-reward-model").to(device)
    print(f'reward model parameters to be updated (ValueHead + 769 params):\n{rl.print_number_of_trainable_model_parameters(toxicity_model)}\n')
    print(toxicity_model.config.id2label)
    non_toxic_text = "#Person 1# tells Tommy that he didn't like the movie."

    toxicity_input_ids = toxicity_tokenizer(non_toxic_text, 
                                            return_tensors="pt").input_ids.to(device)

    logits = toxicity_model(input_ids=toxicity_input_ids).logits
    print(f'logits [not hate, hate]: {logits.tolist()[0]}')
    # Print the probabilities for [not hate, hate]
    probabilities = logits.softmax(dim=-1).tolist()[0]
    print(f'probabilities [not hate, hate]: {probabilities}')

    # get the logits for "not hate" - this is the reward!
    not_hate_index = 0
    nothate_reward = (logits[:, not_hate_index]).tolist()
    print(f'reward (high): {nothate_reward}')

    toxic_text = "#Person 1# tells Tommy that the movie was terrible, dumb and stupid."

    toxicity_input_ids = toxicity_tokenizer(toxic_text, return_tensors="pt").input_ids.to(device)

    logits = toxicity_model(toxicity_input_ids).logits
    print(f'logits [not hate, hate]: {logits.tolist()[0]}')

    # Print the probabilities for [not hate, hate]
    probabilities = logits.softmax(dim=-1).tolist()[0]
    print(f'probabilities [not hate, hate]: {probabilities}')

    # Get the logits for "not hate" - this is the reward!
    nothate_reward = (logits[:, not_hate_index]).tolist() 
    print(f'reward (low): {nothate_reward}')

    device =  torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    sentiment_pipe = pipeline("sentiment-analysis", 
                            model=toxicity_model_name, 
                            device=device,
                            framework="pt")

    reward_logits_kwargs = {
        "top_k": None, # Return all scores.
        "function_to_apply": "none", # Set to "none" to retrieve raw logits.
        "batch_size": 16
    }

    reward_probabilities_kwargs = {
        "top_k": None, # Return all scores.
        "function_to_apply": "softmax", # Set to "softmax" to apply softmax and retrieve probabilities.
        "batch_size": 16
    }

    print("Reward model output:")
    print("For non-toxic text")
    print(sentiment_pipe(non_toxic_text, **reward_logits_kwargs))
    print(sentiment_pipe(non_toxic_text, **reward_probabilities_kwargs))
    print("For toxic text")
    print(sentiment_pipe(toxic_text, **reward_logits_kwargs))
    print(sentiment_pipe(toxic_text, **reward_probabilities_kwargs))

    print(sentiment_pipe(non_toxic_text, **reward_logits_kwargs))
    print(sentiment_pipe(non_toxic_text, **reward_probabilities_kwargs))

    print(sentiment_pipe(toxic_text, **reward_logits_kwargs))
    print(sentiment_pipe(toxic_text, **reward_probabilities_kwargs))












    toxicity_evaluator = evaluate.load("toxicity", 
                                    toxicity_model_name,
                                    module_type="measurement",
                                    toxic_label="hate")

    toxicity_score = toxicity_evaluator.compute(predictions=[
    non_toxic_text
    ])

    print("sentence : ", non_toxic_text)
    print("Toxicity score for non-toxic text:")
    print(toxicity_score["toxicity"])

    toxicity_score = toxicity_evaluator.compute(predictions=[
        toxic_text
    ])

    print("sentence : ", toxic_text)
    print("\nToxicity score for toxic text:")
    print(toxicity_score["toxicity"])











    # Evaluation
    tokenizer = AutoTokenizer.from_pretrained("saved_models/ppo_model", device_map="auto")

    mean_before_detoxification, std_before_detoxification = Eval.evaluate_toxicity(model=ref_model, 
                                                                          toxicity_evaluator=toxicity_evaluator, 
                                                                          tokenizer=tokenizer, 
                                                                          dataset=dataset["test"], 
                                                                          num_samples=10)

    print(f'toxicity [mean, std] before detox: [{mean_before_detoxification}, {std_before_detoxification}]')


    mean_after_detoxification, std_after_detoxification = Eval.evaluate_toxicity(model=ppo_model, 
                                                                    toxicity_evaluator=toxicity_evaluator, 
                                                                    tokenizer=tokenizer, 
                                                                    dataset=dataset["test"], 
                                                                    num_samples=10)
    
    print(f'toxicity [mean, std] after detox: [{mean_after_detoxification}, {std_after_detoxification}]')

    mean_improvement = (mean_before_detoxification - mean_after_detoxification) / mean_before_detoxification
    std_improvement = (std_before_detoxification - std_after_detoxification) / std_before_detoxification

    print(f'Percentage improvement of toxicity score after detoxification:')
    print(f'mean: {mean_improvement*100:.2f}%')
    print(f'std: {std_improvement*100:.2f}%')
if __name__ == "__main__":
    main()
