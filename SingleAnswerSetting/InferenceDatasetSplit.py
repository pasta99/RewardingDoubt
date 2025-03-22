from unsloth import FastLanguageModel
from torch.utils.data import DataLoader
import torch
from tqdm.auto import tqdm
import argparse
from util.ModelLoader import load_model_tokenizer

from util.DataHelper import DataCollatorForTokenizedQueries, load_prepared_dataset
from util.ResponseHandling import response_to_QAResult, save_QAResults

def inference_dataset(model_dir: str, is_unsloth_model: bool, dataset: str, split: str, out_dir: str, batchsize: int = 32) -> None:
    # Load model and tokenizer
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model, tokenizer = load_model_tokenizer(model_dir, is_unsloth_model, device)

    # Load dataset
    dataset = load_prepared_dataset(dataset, split, "verbalize", tokenizer)
    data_collator = DataCollatorForTokenizedQueries(tokenizer)
    dataloader = DataLoader(dataset, batch_size=batchsize, collate_fn=data_collator)

    # Prepare generation
    generation_kwargs_prediction = {
        "max_new_tokens": 32,
        "eos_token_id": [tokenizer.eos_token_id,tokenizer.convert_tokens_to_ids("<|eot_id|>")],
        "do_sample": True,
        "temperature": 0.6,
        "top_p": 0.9,
        "pad_token_id": tokenizer.eos_token_id
    }

    if is_unsloth_model:
        FastLanguageModel.for_inference(model)
    else:
        model.eval()

    prediction_results = []
    progress_bar = tqdm(range(len(dataloader)))
    for idx, batch in enumerate(dataloader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        gt_candidates = batch["gt_candidates"]
        questions = batch["question"]
        is_multiple_choice = batch["is_multiple_choice"]
    
        prediction = model.generate(input_ids=input_ids, attention_mask=attention_mask, **generation_kwargs_prediction)

        responses = [prediction[i][len(input_ids[i]):] for i in range(len(prediction))] # Cutoff results so only newly generated tokens are left over

        responses_decoded = tokenizer.batch_decode(responses, skip_special_tokens=True)

        results = [response_to_QAResult(question, response, gt, is_mc) for question, response, gt, is_mc in zip(questions, responses_decoded, gt_candidates, is_multiple_choice)]
        prediction_results += results

        progress_bar.update(1)

    save_QAResults(prediction_results, out_dir)

def setup_parser():
    parser = argparse.ArgumentParser(description="Generate responses for whole dataset split and save to file")
    parser.add_argument("--model_dir", type=str, default="unsloth/llama-3-8b-Instruct-bnb-4bit")
    parser.add_argument("--out_dir", type=str, default="outputs/debug/inference.json")
    parser.add_argument("--dataset", type=str, default="triviaqa", choices=["triviaqa", "medqa", "commonsenseqa"])
    parser.add_argument("--batchsize", type=int, default=32)
    parser.add_argument("--is_unsloth", action='store_true')
    parser.add_argument("--split",  type=str, default="validation")

    return parser


if __name__ == "__main__":
    parser = setup_parser()
    args = parser.parse_args()
    inference_dataset(args.model_dir, args.is_unsloth, args.dataset, args.split, args.out_dir, args.batchsize)