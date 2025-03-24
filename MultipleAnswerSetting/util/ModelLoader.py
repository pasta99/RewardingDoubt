from unsloth import FastLanguageModel
from trl import AutoModelForCausalLMWithValueHead
import trl
from peft import LoraConfig
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def get_device():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    return device

def load_lora_model_tokenizer(model_dir, is_unsloth, device):
    if is_unsloth:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = model_dir,
            max_seq_length = 1048,
            dtype = None,
            load_in_4bit = True,
        )
        model = FastLanguageModel.get_peft_model(
            model,
            r = 8,
            # target_modules = ["q_proj", "k_proj"],
            lora_alpha = 8,
            lora_dropout = 0, 
            bias = "none",
            use_gradient_checkpointing = "unsloth", 
            random_state = 3407,
            use_rslora = False,  
            loftq_config = None
        )

        model = AutoModelForCausalLMWithValueHead.from_pretrained(model)
        trl.trainer.peft_module_casting_to_bf16(model)
    
    else:
        model_id = model_dir
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        lora_config = LoraConfig(
                                r=16,
                                lora_alpha=32,
                                lora_dropout=0.05,
                                bias="none",
                                task_type="CAUSAL_LM"
                                )
        model = AutoModelForCausalLMWithValueHead.from_pretrained(model, peft_config=lora_config)
        model.to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_dir)

    tokenizer.pad_token_id = tokenizer.eos_token_id
    model.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    model.padding_side='left'

    return model, tokenizer

def load_model_tokenizer(model_dir, is_unsloth, device):
    if is_unsloth:
        model, tokenizer = FastLanguageModel.from_pretrained(
                model_name = model_dir,
                max_seq_length = 1048,
                dtype = None,
                load_in_4bit = True,
        )
    else:
        model_id = model_dir
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    tokenizer.pad_token_id = tokenizer.eos_token_id
    model.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    model.padding_side='left'

    return model, tokenizer

def load_tokenizer(dir):
    tokenizer = AutoTokenizer.from_pretrained(dir)

    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    return tokenizer