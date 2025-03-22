import argparse
import torch
import os
import numpy as np
import wandb

from trl import PPOConfig
from trl import AutoModelForCausalLMWithValueHead
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from util.DataHelper import DataCollatorForTokenizedQueries
from unsloth import FastLanguageModel
from util.ppo_trainer_no_cache import PPOTrainerNoCache
from transformers import AutoTokenizer
from typing import Tuple

from util.DataHelper import load_prepared_dataset
from util.ResponseHandling import response_to_QAResult
from util.RLHelper import QAResult_to_reward
from util.util import remove_padding
from util.ModelLoader import load_lora_model_tokenizer, load_model_tokenizer

torch.manual_seed(2)

def evaluate_model(model, dataloader, tokenizer, generation_kwargs, device) -> Tuple[float, float]:
    rewards_epoch = []

    for idx, batch in enumerate(dataloader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        gt_candidates = batch["gt_candidates"]
        questions = batch["question"]
        is_multiple_choice = batch["is_multiple_choice"]
    
        prediction = model.generate(input_ids=input_ids, attention_mask=attention_mask, **generation_kwargs)

        responses = [prediction[i][len(input_ids[i]):] for i in range(len(prediction))] # Cutoff results so only newly generated tokens are left over

        responses_decoded = tokenizer.batch_decode(responses, skip_special_tokens=True)

        results = [response_to_QAResult(question, response, gt, is_mc) for question, response, gt, is_mc in zip(questions, responses_decoded, gt_candidates, is_multiple_choice)]
        rewards = [QAResult_to_reward(r) for r in results]
        rewards_epoch += rewards
    
    return np.mean(rewards_epoch), np.std(rewards_epoch)

def train(out_dir: str, lr: float, epochs: int, batchsize: int, model_dir: str, tokenizer_dir: str, dataset: str, log_with: str = "tensorboard", is_unsloth: bool = True, use_lora: bool = True) -> None: 
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    # Set up model and tokenizer
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if use_lora:
        model, tokenizer = load_lora_model_tokenizer(model_dir, is_unsloth, device)
    else:
        model, tokenizer = load_model_tokenizer(model_dir, is_unsloth, device)

    tokenizer.pad_token_id = tokenizer.eos_token_id
    model.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    model.padding_side='left'
    # model = AutoModelForCausalLMWithValueHead.from_pretrained(model)
    
    # Set up datsets and loader
    dataset_train = load_prepared_dataset(dataset, "train", "verbalize", tokenizer)
    dataset_validation = load_prepared_dataset(dataset, "validation", "verbalize", tokenizer)

    pad_tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    pad_tokenizer.pad_token_id = tokenizer.eos_token_id
    pad_tokenizer.padding_side = "left"

    data_collator = DataCollatorForTokenizedQueries(pad_tokenizer)
    dataloader_validation = DataLoader(dataset_validation, batch_size=2*batchsize, collate_fn=data_collator)

    # Set up PPO Trainer
    if is_unsloth:
        FastLanguageModel.for_inference(model.pretrained_model)

    config = PPOConfig(
        learning_rate=lr,
        task_name="gpt", 
        batch_size=batchsize, 
        mini_batch_size=int(batchsize/2), 
        log_with=log_with, 
        project_kwargs={"logging_dir": out_dir},
        remove_unused_columns=False,
        optimize_device_cache=True,
        init_kl_coef=0.05,
        
    )

    ppo_trainer = PPOTrainerNoCache(
        model=model,
        config=config,
        dataset=dataset_train,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    prediction_terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        tokenizer.convert_tokens_to_ids("ĠConfidence")
    ]

    ppo_terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>"),
    ]

    generation_kwargs_prediction = {
        "max_new_tokens": 256,
        "eos_token_id": prediction_terminators,
        "do_sample": True,
        "temperature": 0.6,
        "top_p": 0.9,
        "pad_token_id": tokenizer.eos_token_id
    }

    generation_kwargs_ppo = {
        "min_length": -1, # don't ignore the EOS token (see above)
        "top_k": 0.0, # no top-k sampling
        "top_p": 1.0, # no nucleus sampling
        "do_sample": True, # yes, we want to sample
        "pad_token_id": tokenizer.eos_token_id, # most decoder models don't have a padding token - use EOS token instead
        "max_new_tokens": 32,
        "eos_token_id": ppo_terminators,
        "max_new_tokens": 500
    }

    eval_terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>"),
    ]

    generation_kwargs_eval = {
        "max_new_tokens": 256,
        "eos_token_id": eval_terminators,
        "do_sample": True,
        "temperature": 0.6,
        "top_p": 0.9,
        "pad_token_id": tokenizer.eos_token_id
    }

    best_reward = -100
    best_reward_epoch = -1

    for epoch in range(epochs):

        rewards_epoch = []
        for idx, batch in enumerate(ppo_trainer.dataloader):
            
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            gt_candidates = batch["gt_candidates"]
            questions = batch["question"]
            is_multiple_choice = batch["is_multiple_choice"]

            if is_unsloth:
                FastLanguageModel.for_inference(model.pretrained_model)
            else:
                model.eval()

            prediction = model.generate(input_ids=input_ids, attention_mask=attention_mask, **generation_kwargs_prediction)
            prediction = [remove_padding(p, tokenizer.pad_token_id) for p in prediction]

            # Generate confidence
            if not is_unsloth:
                model.train()
            response_tensors = ppo_trainer.generate(prediction, return_prompt=False, **generation_kwargs_ppo)

            # Create prediction + confidence output
            total_tensor = [torch.cat((p, c), 0) for p, c in zip(prediction, response_tensors)]
            answer_only_tensor = [total_tensor[i][len(input_ids[i]):] for i in range(len(input_ids))]

            responses_decoded = tokenizer.batch_decode(answer_only_tensor, skip_special_tokens=True)
            
            # Parse prediction and confidence
            results = [response_to_QAResult(question, response, gt, is_mc) for question, response, gt, is_mc in zip(questions, responses_decoded, gt_candidates, is_multiple_choice)]

            # Compute rewards
            rewards = [QAResult_to_reward(r) for r in results]
            rewards_epoch += rewards
            rewards = [torch.tensor(r).to(device) for r in rewards]

            # Create log data
            batch["response"] = responses_decoded
            batch["query"] = batch["question"]

            try:
                if is_unsloth:
                    FastLanguageModel.for_training(model.pretrained_model)
                stats = ppo_trainer.step(prediction, response_tensors, rewards)
            except IndexError:
                print(f"INDEX ERROR detected and ignored ({idx})")

            ppo_trainer.log_stats(stats, batch, rewards, columns_to_log=["query", "response", "answer"])
        
        avg_reward = np.mean(rewards_epoch)
        
        print(f"Finished epoch {epoch}. Average reward: {avg_reward}")
        ppo_trainer.save_pretrained(os.path.join(out_dir, "model_finetuned"))

        # Evaluate model after each epoch
        if is_unsloth:
            FastLanguageModel.for_inference(model.pretrained_model)
        else:
            model.eval()
        mean_reward, std_reward = evaluate_model(model, dataloader_validation, tokenizer, generation_kwargs_eval, device)

        if log_with == "wandb":
            wandb.log({"mean_reward_evaluation": mean_reward})
            wandb.log({"std_reward_evaluation": std_reward})

        # Save the best performing model
        mean_reward = avg_reward
        if mean_reward > best_reward: 
            ppo_trainer.save_pretrained(os.path.join(out_dir, "model_finetuned_best"))
            best_reward = mean_reward
            best_reward_epoch = epoch 

    print("Finished Training!")
    print(f"Best avg reward {best_reward} in epoch {best_reward_epoch}")
    return

def setup_parser():
    parser = argparse.ArgumentParser(description="Training an example model")
    parser.add_argument("--model_dir", type=str, default="unsloth/llama-3-8b-Instruct-bnb-4bit")
    parser.add_argument("--tokenizer_dir", type=str, default="unsloth/llama-3-8b-Instruct-bnb-4bit")
    parser.add_argument("--out_dir", type=str, default="outputs/debug")
    parser.add_argument("--lr", type=float, default=1.41e-5)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batchsize", type=int, default=2)
    parser.add_argument("--log_with", type=str, default="tensorboard")
    parser.add_argument("--dataset", type=str, default="medqa")
    parser.add_argument("--is_unsloth", action='store_true')

    return parser


if __name__ == "__main__":
    parser = setup_parser()
    args = parser.parse_args()
    train(args.out_dir, lr=args.lr, epochs=args.epochs, batchsize=args.batchsize, model_dir=args.model_dir, tokenizer_dir=args.tokenizer_dir, dataset=args.dataset, log_with=args.log_with, is_unsloth=args.is_unsloth)
