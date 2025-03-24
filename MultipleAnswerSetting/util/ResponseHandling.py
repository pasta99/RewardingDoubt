import dataclasses
from typing import List, Optional
import re
import torch
import json
from dacite import from_dict as dict_to_class
import random

from util.MultiFactResult import MultiFactResult

from util.RLHelper import reward_function
from util.Metrics import are_answers_correct, Metric

def MultiFactResult_to_dict(result: MultiFactResult) -> dict:
    return dataclasses.asdict(result)

def dict_to_MultiFactResult(result: dict) -> MultiFactResult:
    return dict_to_class(MultiFactResult, result)

def split_answers(response):
    pattern = r"Answer: (.*?), Confidence: ([0-9.]+)"
    matches = re.findall(pattern, response)

    results = []
    for answer, confidence in matches:
        try: 
            parsed_confidence = int(confidence)
        except ValueError:
            parsed_confidence = -1
        fact = {"answer": answer, "confidence": parsed_confidence} 
        results.append(fact)

    return results

def response_to_MultiFactResult(question, response, gt):
    results = split_answers(response)

    predictions = [r["answer"] for r in results]
    confidences = [r["confidence"] for r in results]

    res = MultiFactResult(question, response, predictions, confidences, gt)

    return res

def save_MultiFactResult(results: List[MultiFactResult], out_dir: str):
    results_dict = [MultiFactResult_to_dict(res) for res in results]

    with open(out_dir, 'w') as fout:
        json.dump(results_dict , fout, indent=4)

def load_MultiFactResults(read_dir: str) -> List[MultiFactResult]:
    with open(read_dir, "r") as f:
        results_dicts = json.load(f) 

    results = [dict_to_MultiFactResult(r) for r in results_dicts]
    return results

def MultiFactResult_to_answers(result: MultiFactResult, original_query, tokenizer):
    generated = [f"Answer: {answer}, Confidence: " for answer, confidence in zip(result.predictions, result.confidences)]
    answers_correct = are_answers_correct(result.predictions, result.gt_candidates, Metric.F1, 0.5)
    
    inputs = []
    responses = []
    prev_text_ids = original_query.cpu()
    for predicted, confidence in zip(generated, result.confidences):
        prediction_ids = tokenizer.encode(predicted, return_tensors="pt").cpu()
        input = torch.cat((prev_text_ids, prediction_ids[0][1:]))
        inputs.append(input)
        response = tokenizer.encode(f"{int(confidence)}\n", return_tensors="pt")[0][1:].cpu()
        responses.append(response)
        prev_text_ids = torch.cat((input, response))
        # break
    
    if len(responses) > 0:
        if len(responses[0]) > 0:
            responses[-1][-1] = tokenizer.eos_token_id

    return inputs, answers_correct

def batch_MultiFactResult_to_answers(batch_result: List[MultiFactResult], batch_original_query, tokenizer, device):
    batched = [MultiFactResult_to_answers(r, i, tokenizer) for r, i in zip(batch_result, batch_original_query)] 
    batch_inputs, batch_factual = map(list, zip(*batched))
    return batch_inputs, batch_factual

def MultiFactResult_to_reinforcementsteps(result: MultiFactResult, original_query, tokenizer, device):
    generated = [f"Answer: {answer}, Confidence: " for answer, confidence in zip(result.predictions, result.confidences)]
    answers_correct = are_answers_correct(result.predictions, result.gt_candidates, Metric.F1, 0.5)
    rewards = [reward_function(c, a_correct) for c, a_correct in zip(result.confidences, answers_correct)]
   
    inputs = []
    responses = []
    prev_text_ids = original_query.cpu()
    for predicted, confidence in zip(generated, result.confidences):
        prediction_ids = tokenizer.encode(predicted, return_tensors="pt").cpu()
        input = torch.cat((prev_text_ids, prediction_ids[0][1:]))
        inputs.append(input)
        response = tokenizer.encode(f"{int(confidence)}\n", return_tensors="pt")[0][1:].cpu()
        responses.append(response)
        prev_text_ids = torch.cat((input, response))
        # break
    
    if len(responses) > 0:
        if len(responses[0]) > 0:
            responses[-1][-1] = tokenizer.eos_token_id

    return inputs, responses, rewards

def batch_MultiFactResult_to_reinforcementsteps(batch_result: List[MultiFactResult], batch_original_query, tokenizer, device):
    batched = [MultiFactResult_to_reinforcementsteps(r, i, tokenizer, device) for r, i in zip(batch_result, batch_original_query)] 
    batch_inputs, batch_responses, batch_rewards = map(list, zip(*batched))
    return batch_inputs, batch_responses, batch_rewards

def sort_steps(inputs, responses, rewards, questions, answers, sort_index = 0):
    combined = list(zip(inputs, responses, rewards, questions, answers))
    sorted_combined = sorted(combined, key=lambda x: len(x[sort_index]))
    sorted_inputs, sorted_responses, sorted_rewards, sorted_questions, sorted_answers = map(list, zip(*sorted_combined))
    return sorted_inputs, sorted_responses, sorted_rewards, sorted_questions, sorted_answers

def sort_together(*lists, sort_index = 0):
    combined = list(zip(lists))
    sorted_combined = sorted(combined, key=lambda x: len(x[sort_index]))

    return map(list, zip(*sorted_combined))

def shuffle_lists_in_same_manner(*lists):
    # Ensure all lists have the same length
    if not all(len(lst) == len(lists[0]) for lst in lists):
        raise ValueError("All lists must have the same length")
    
    # Combine the lists into a list of tuples
    combined = list(zip(*lists))
    
    # Shuffle the combined list
    random.shuffle(combined)
    
    # Unzip the shuffled list back into individual lists
    return map(list, zip(*combined))

def split_into_chunks(data, chunk_size):
    return [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]

def batch_data(inputs_flat, responses_flat, rewards_flat, questions_flat, answers_flat, batchsize):
    inputs = split_into_chunks(inputs_flat, batchsize)
    responses = split_into_chunks(responses_flat, batchsize)
    rewards = split_into_chunks(rewards_flat, batchsize)
    questions = split_into_chunks(questions_flat, batchsize)
    answers = split_into_chunks(answers_flat, batchsize)

    return inputs, responses, rewards, questions, answers
