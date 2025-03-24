import argparse
import torch
import os
import numpy as np
import wandb
import itertools
from typing import Tuple

from trl import PPOConfig
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from unsloth import FastLanguageModel
from util.ppo_trainer_no_cache import PPOTrainerNoCache
from transformers import AutoTokenizer

from util.DataHelper import load_prepared_dataset, DataCollatorForTokenizedQueries
from util.ResponseHandling import response_to_MultiFactResult, batch_MultiFactResult_to_reinforcementsteps, shuffle_lists_in_same_manner, batch_MultiFactResult_to_answers, split_into_chunks
from util.ModelLoader import load_lora_model_tokenizer, load_model_tokenizer
from util.RLHelper import reward_function

torch.manual_seed(2)

def parse_to_float(str):
    try:
        return float(str)
    except:
        return -1.0

def evaluate_model(model, dataloader, tokenizer, generation_kwargs, device) -> Tuple[float, float]:
    rewards_epoch = []
    print("Starting Evaluation")

    for idx, batch in enumerate(dataloader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        gt_candidates = batch["gt_candidates"]
        questions = batch["question"]
    
        prediction = model.generate(input_ids=input_ids, attention_mask=attention_mask, **generation_kwargs)

        responses = [prediction[i][len(input_ids[i]):] for i in range(len(prediction))] # Cutoff results so only newly generated tokens are left over

        responses_decoded = tokenizer.batch_decode(responses, skip_special_tokens=True)

        multifactresults = [response_to_MultiFactResult(q, r, g) for q, r, g in zip(questions, responses_decoded, gt_candidates)]
        _, _, batch_rewards = batch_MultiFactResult_to_reinforcementsteps(multifactresults, input_ids, tokenizer, device)
        
        rewards_flat = list(itertools.chain.from_iterable(batch_rewards))

        rewards_epoch += rewards_flat
    
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

    # Set up datsets and loader
    dataset_train = load_prepared_dataset(dataset, "train", tokenizer)
    dataset_validation = load_prepared_dataset(dataset, "validation", tokenizer)

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
        task_name="multifact", 
        batch_size=batchsize, 
        mini_batch_size=int(batchsize/2), 
        log_with=log_with, 
        project_kwargs={"logging_dir": out_dir},
        remove_unused_columns=False,
        optimize_device_cache=True,
        init_kl_coef=0.05
        # kl_penalty="full"
    )

    ppo_trainer = PPOTrainerNoCache(
        model=model,
        config=config,
        dataset=dataset_train,
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>"),
    ]

    generation_kwargs_prediction = {
        "max_new_tokens": 2048,
        "eos_token_id": terminators,
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
        "max_new_tokens": 2,
        "eos_token_id": terminators,
    }

    eval_terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>"),
    ]

    generation_kwargs_eval = {
        "max_new_tokens": 2,
        "eos_token_id": eval_terminators,
        "do_sample": True,
        "temperature": 0.6,
        "top_p": 0.9,
        "pad_token_id": tokenizer.eos_token_id
    }

    best_reward = -100
    best_reward_epoch = -1

    save_interval = 2000

    rest_inputs, rest_factual, rest_gt = [], [], []
    for epoch in range(epochs):

        rewards_epoch = []
        for idx, batch in enumerate(ppo_trainer.dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            gt_candidates = batch["gt_candidates"]
            questions = batch["question"]

            if is_unsloth:
                FastLanguageModel.for_inference(model.pretrained_model)
            else:
                model.eval()

            prediction = model.generate(input_ids=input_ids, attention_mask=attention_mask, **generation_kwargs_prediction)

            prediction = prediction.cpu()

            responses = [prediction[i][len(input_ids[i]):] for i in range(len(prediction))] # Cutoff results so only newly generated tokens are left over

            responses_decoded = tokenizer.batch_decode(responses, skip_special_tokens=True)
            
            multifactresults = [response_to_MultiFactResult(q, r, g) for q, r, g in zip(questions, responses_decoded, gt_candidates)]

            batch_inputs, batch_factual = batch_MultiFactResult_to_answers(multifactresults, input_ids, tokenizer, device)

            batch_gt = []
            for i, r in enumerate(batch_inputs):
                a = []
                for _ in range(len(r)):
                    a.append(batch["gt_candidates"][i])
                batch_gt.append(a)

            # Turn 2D lists into flattened list
            inputs_flat = list(itertools.chain.from_iterable(batch_inputs))
            gt_flat = list(itertools.chain.from_iterable(batch_gt))
            factual_flat = list(itertools.chain.from_iterable(batch_factual))

            # Use leftover statements from last batch
            inputs_flat = rest_inputs + inputs_flat
            rest_inputs = []  # Clear leftovers

            gt_flat = rest_gt + gt_flat
            rest_gt = []

            factual_flat = rest_factual + factual_flat
            rest_factual = []

            if len(inputs_flat) == 0 or len(gt_flat) == 0 or len(factual_flat) == 0:
                continue

            # Shuffle lists so model gets more than one question per batch
            inputs_flat, gt_flat, factual_flat = shuffle_lists_in_same_manner(inputs_flat, gt_flat, factual_flat)              

            # Turn all statements in subbatches of size <batchsize>
            batch_input_ids = split_into_chunks(inputs_flat, batchsize)
            batch_gt = split_into_chunks(gt_flat, batchsize)
            batch_factual = split_into_chunks(factual_flat, batchsize)

            # Train for all subbatches
            for input, gt, factual in zip(batch_input_ids, batch_gt, batch_factual):
                if len(input) == batchsize:
                    prediction = tokenizer.batch_decode(input)
                    
                    if is_unsloth:
                        FastLanguageModel.for_inference(model.pretrained_model)
                    response_tensors = ppo_trainer.generate(input, return_prompt=False, **generation_kwargs_ppo)

                    confidences = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)
                    confidences = [parse_to_float(c) for c in confidences]

                    rewards = [reward_function(c, f) for c, f in zip(confidences, factual)]

                    # Prepare inputs
                    input = [i.to(device) for i in input]
                    response = [r.to(device) for r in response_tensors]
                    reward = [torch.tensor(r).to(device) for r in rewards]
                    
                    if is_unsloth:
                        FastLanguageModel.for_training(model.pretrained_model)

                    stats = ppo_trainer.step(input, response, reward)

                    # Create log data
                    ppo_batch = {}
                    ppo_batch["response"] = confidences
                    ppo_batch["query"] = prediction
                    ppo_batch["answer"] = gt

                    ppo_trainer.log_stats(stats, ppo_batch, reward, columns_to_log=["query", "response", "answer"])
                else:
                    rest_inputs = input
                    rest_factual = factual
                    rest_gt = gt
                    break
            
            if idx % save_interval == 0:
                ppo_trainer.save_pretrained(os.path.join(out_dir, f"model_finetuned{epoch}_{idx}"))

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
    parser.add_argument("--batchsize", type=int, default=4)
    parser.add_argument("--log_with", type=str, default="tensorboard")
    parser.add_argument("--dataset", type=str, default="qampari")
    parser.add_argument("--is_unsloth", action='store_true')

    return parser


if __name__ == "__main__":
    parser = setup_parser()
    args = parser.parse_args()
    train(args.out_dir, lr=args.lr, epochs=args.epochs, batchsize=args.batchsize, model_dir=args.model_dir, tokenizer_dir=args.tokenizer_dir, dataset=args.dataset, log_with=args.log_with, is_unsloth=args.is_unsloth)
