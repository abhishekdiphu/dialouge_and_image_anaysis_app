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


    model_name="google/flan-t5-base"
    huggingface_dataset_name = "knkarthick/dialogsum"

    dataset_original = rl.load_data(huggingface_dataset_name)
    #dataset_original.save_to_disk("dataset_rlhf")

    if print_debug:
        print(dataset_original)
    
    dataset = rl.build_dataset(model_name=model_name,
                        dataset_name=huggingface_dataset_name,
                        input_min_text_length=200, 
                        input_max_text_length=1000)
    if print_debug:
        print(dataset)

    lora_config = LoraConfig(
    r=32, # Rank
    lora_alpha=32,
    target_modules=["q", "v"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.SEQ_2_SEQ_LM # FLAN-T5
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, 
                                                torch_dtype=torch.float32)

    peft_model = PeftModel.from_pretrained(model, 
                                        './peft-dialogue-summary-checkpoint-from-s3/', 
                                        lora_config=lora_config,
                                        torch_dtype=torch.float32, 
                                        device_map="auto",                                       
                                        is_trainable=True)

    print(f'PEFT model parameters to be updated:\n{rl.print_number_of_trainable_model_parameters(peft_model)}\n')
    
    

    ppo_model = AutoModelForSeq2SeqLMWithValueHead.from_pretrained(peft_model,                                                               
                                                               torch_dtype=torch.float32,
                                                               is_trainable=True).to(device)

    print(f'PPO model parameters to be updated (ValueHead + 769 params):\n{rl.print_number_of_trainable_model_parameters(ppo_model)}\n')
    print(ppo_model.v_head)
    ref_model = create_reference_model(ppo_model)

    #save the ref model too.
    ref_save_dir = "./saved_models"
    ref_model_dir = os.path.join(ref_save_dir, "ref_model")
    os.makedirs(ref_model_dir, exist_ok=True)
    ppo_model.pretrained_model.save_pretrained(ref_model_dir)

    print(f'Reference model parameters to be updated:\n{rl.print_number_of_trainable_model_parameters(ref_model)}\n')



    # REWARD MODEL CLASSIFer
    toxicity_model_name = "facebook/roberta-hate-speech-dynabench-r4-target"
    toxicity_tokenizer = AutoTokenizer.from_pretrained(toxicity_model_name)
    toxicity_model = AutoModelForSequenceClassification.from_pretrained(
                    toxicity_model_name).to(device)
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
    tokenizer = AutoTokenizer.from_pretrained(model_name, device_map="auto")

    mean_before_detoxification, std_before_detoxification = Eval.evaluate_toxicity(model=ref_model, 
                                                                          toxicity_evaluator=toxicity_evaluator, 
                                                                          tokenizer=tokenizer, 
                                                                          dataset=dataset["test"], 
                                                                          num_samples=10)

    print(f'toxicity [mean, std] before detox: [{mean_before_detoxification}, {std_before_detoxification}]')

    learning_rate=1.41e-5
    max_ppo_epochs=1
    mini_batch_size=4
    batch_size=16

    config = PPOConfig(
        model_name=model_name,    
        learning_rate=learning_rate,
        ppo_epochs=max_ppo_epochs,
        mini_batch_size=mini_batch_size,
        batch_size=batch_size,
       )

    ppo_trainer = PPOTrainer(config=config, 
                            model=ppo_model, 
                            ref_model=ref_model, 
                            tokenizer=tokenizer, 
                            dataset=dataset["train"], 
                            data_collator=rl.collator)
    output_min_length = 100
    output_max_length = 400
    output_length_sampler = LengthSampler(output_min_length, output_max_length)

    generation_kwargs = {
        "min_length": 5,
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True
    }

    reward_kwargs = {
        "top_k": None, # Return all scores.
        "function_to_apply": "none", # You want the raw logits without softmax.
        "batch_size": 16
    }

    max_ppo_steps = 2 

    for step, batch in tqdm(enumerate(ppo_trainer.dataloader)):
        # Break when you reach max_steps.
        if step >= max_ppo_steps:
            break   

        prompt_tensors = batch["input_ids"]

        # Get response from FLAN-T5/PEFT LLM.
        summary_tensors = []

        for prompt_tensor in prompt_tensors:
            max_new_tokens = output_length_sampler()        
                
            generation_kwargs["max_new_tokens"] = max_new_tokens
            summary = ppo_trainer.generate(prompt_tensor, **generation_kwargs)
            
            summary_tensors.append(summary.squeeze()[-max_new_tokens:].to(device))
            
        # This needs to be called "response".
        batch["response"] = [tokenizer.decode(r.squeeze()) for r in summary_tensors]

        # Compute reward outputs.
        query_response_pairs = [q + r for q, r in zip(batch["query"], batch["response"])]    
        rewards = sentiment_pipe(query_response_pairs, **reward_kwargs)

        # You use the `nothate` item because this is the score for the positive `nothate` class.
        reward_tensors = [torch.tensor(reward[not_hate_index]["score"]) for reward in rewards]    
        

        print("_____________", type(reward_tensors))
        print("_____________", type(prompt_tensors))
        print("_____________", type(summary_tensors))
        # Run PPO step.
        stats = ppo_trainer.step(prompt_tensors, summary_tensors, reward_tensors)
        ppo_trainer.log_stats(stats, batch, reward_tensors)
        
        print(f'objective/kl: {stats["objective/kl"]}')
        print(f'ppo/returns/mean: {stats["ppo/returns/mean"]}')
        print(f'ppo/policy/advantages_mean: {stats["ppo/policy/advantages_mean"]}')
        print('-'.join('' for x in range(100)))




    
    
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

    save_all_models = True
    if save_all_models:
        save_dir = "./saved_models"
        os.makedirs(save_dir, exist_ok=True) 

        print(".....................Saving models.......")
        # Save the trained PPO model.
        ppo_model_dir = os.path.join(save_dir, "ppo_model")
        os.makedirs(ppo_model_dir, exist_ok=True)
        
        ppo_trainer.model.save_pretrained(ppo_model_dir)
        # Save the tokenizer too.
        tokenizer.save_pretrained(ppo_model_dir)


        # save reward model
        reward_model_dir = os.path.join(save_dir, "toxicity-reward-model")
        os.makedirs(ref_model_dir, exist_ok=True)
        
        toxicity_model.save_pretrained(reward_model_dir)

        toxicity_tokenizer.save_pretrained(reward_model_dir)
        print(".....................Models saved.......")
if __name__ == "__main__":
    main()
