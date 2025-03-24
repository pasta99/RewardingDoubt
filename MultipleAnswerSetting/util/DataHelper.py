from transformers.data.data_collator import DataCollatorMixin
from datasets import load_dataset
import numpy as np
from dataclasses import dataclass
from collections.abc import Callable
from typing import List
from util.Prompts import get_prompt

@dataclass
class DatasetDescriptor:
    huggingface_config: str
    normalize_function: Callable
    type: str
    columns_to_remove: List[str]

class DataCollatorForTokenizedQueries(DataCollatorMixin):
    def __init__(self, tokenizer, padding=True, max_length = None):
        self.tokenizer = tokenizer
        self.padding = padding
        self.max_length = max_length

    def __call__(self, batch):
        queries = [sample["query"] for sample in batch]
        
        # Pad the "queries"
        tokenized_queries = self.tokenizer.pad(
            {"input_ids": queries},
            padding=self.padding,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # Create a new dictionary for the collated batch
        collated_batch = {
            "input_ids": tokenized_queries["input_ids"],
            "attention_mask": tokenized_queries["attention_mask"]
        }
        
        for key in batch[0].keys():
            collated_batch[key] = [sample[key] for sample in batch]

        del collated_batch["query"]

        return collated_batch
    
def prepare_queries(sample, tokenizer, systemprompt, tokenize=False):
    question = sample["question"]
    # The first answer is considered the ground truth
    gt_candidates = sample["gt_candidates"]

    messages = [
            {"role": "system", "content": systemprompt},
            {"role": "user", "content": question}
        ]
    query = tokenizer.apply_chat_template(
                    messages, 
                    tokenize=tokenize, 
                    add_generation_prompt=True
            )

    sample_new = {"query": query, "gt_candidates": gt_candidates}
    return sample_new

def _normalize_qampari(sample):
    question = sample["question"]
    answer = ", ".join(sample["answers"])
    answer_aliases = []
    for a in sample["answer_list"]["aliases"]:
        answer_aliases += a

    return {"question": question, "answer": answer, "gt_candidates": answer_aliases, "is_multiple_choise":False}

### ADD NORMALIZE FUNCTION HERE:


datasets = {
    "qampari": DatasetDescriptor
    (
        huggingface_config = {"path": "iohadrubin/qampari", "name": "reranking_bm25"}, 
        normalize_function = _normalize_qampari, 
        type = "multifact", 
        columns_to_remove = ['question_text', 'answer_list', 'qid', 'question', 'answers', 'positive_ctxs', 'hard_negative_ctxs']
    ),
    ### ADD DATASET DESCRIPTOR HERE:
}

def get_dataset_descriptor(dataset):
    try:
        dataset_desc = datasets[dataset]
    except KeyError:
        raise NotImplementedError(f"This dataset has not been implemented: '{dataset}'")
    
    return dataset_desc

def load_prepared_dataset(dataset, split, tokenizer):
    dataset_descriptor = get_dataset_descriptor(dataset)

    dataset = load_dataset(**dataset_descriptor.huggingface_config, split=split)
    dataset_norm = dataset.map(lambda x: dataset_descriptor.normalize_function(x), remove_columns=dataset_descriptor.columns_to_remove)
    system_prompt = get_prompt(dataset_descriptor.type)
    dataset = dataset_norm.map(lambda x: prepare_queries(x, tokenizer, system_prompt, tokenize=True))

    return dataset