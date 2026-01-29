# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import os

# import fire
import torch
# import torch.distributed as dist
import torch.optim as optim
from peft import get_peft_model, LoraConfig
from pkg_resources import packaging
# from torch.distributed.fsdp import (
#     FullyShardedDataParallel as FSDP,
# )
from torch.optim.lr_scheduler import StepLR
# from torch.utils.data import DistributedSampler
from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer,
    LlamaConfig,
    default_data_collator,
)
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
import datasets
# import policies
# from configs import fsdp_config, train_config
# from configs import train_config
# from policies import AnyPrecisionAdamW

# from utils import fsdp_auto_wrap_policy
# from utils.config_utils import (
#     update_config,
#     generate_peft_config,
#     generate_dataset_config,
# )
# from utils.dataset_utils import get_preprocessed_dataset

from utils.AsFT_train_utils import (
    train,
    # freeze_transformer_layers,
    # setup,
    # setup_environ_flags,
    # clear_gpu_cache,
    # print_model_size,
    # get_policies
)

from utils.dataset_utils import StandardDataset

import copy

import numpy
import torch
from transformers import AutoModelForCausalLM
from box import Box

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--config_path', help='Path to config file')
parser.add_argument('--save_folder', help='Folder to save finetuned model')


def convert_to_template_from_alpaca(example):
    return {"prompt": [
                         {"role": "user", "content": example['instruction'] + ": " + example['input']}],
            "completion": [
                         {"role": "assistant", "content": example['output']}
                        ]
           }


TEMPLATE_FUNC_MAP = {
    'alpaca': convert_to_template_from_alpaca
}

dtype_map = {"float16": torch.float16,
             "bfloat16": torch.bfloat16,
                "float32": torch.float32
}

class SafeLoRA:
    def __init__(self, peft_model: torch.nn.Module, config):
        super().__init__()
        self.peft_model = peft_model
        self.config = config
        self.peft_config = peft_model.peft_config["default"]
        # self.model_ori = copy.deepcopy(peft_model) # not used

        self.project_matrix = self.get_aligned_matrix()

    def get_aligned_matrix(self):

        base_model = AutoModelForCausalLM.from_pretrained(
            self.config.base_model_path,
            return_dict=True,
            device_map="cpu",
            low_cpu_mem_usage=True,
        )
        aligned_model = AutoModelForCausalLM.from_pretrained(
            self.config.aligned_model_path,
            return_dict=True,
            device_map="cpu",
            low_cpu_mem_usage=True,
        )

        v = []
        proj_modules = list(self.peft_config.target_modules)
        for (b_name, b_param), (a_name, a_param) in zip(base_model.named_parameters(), aligned_model.named_parameters()):
            if any(module in a_name for module in proj_modules):
                assert b_param.shape == a_param.shape, "The dimensions of the base model's weight should be the same with the aligned model's weight."
                vec = a_param - b_param
                vec = vec.to(self.config.device)
                vec = torch.mm(vec, vec.t()) / torch.norm(vec)
                v.append(vec.detach().cpu())
        return v


class SafeLoRAConfig:
    def __init__(self, base_model_path, aligned_model_path, device):
        self.base_model_path = base_model_path 
        self.aligned_model_path = aligned_model_path  
        self.device = device  

def main(config_path, save_folder):
    
    
    # Update the configuration for the training and sharding process
    # update_config((train_config, fsdp_config), **kwargs)

    # read our config
    config = Box.from_yaml(filename=config_path)

    safe_config = SafeLoRAConfig(
        base_model_path=config.base_model_path,
        aligned_model_path=config.aligned_model_path,
        device=config.device #either 'cuda' or 'cpu'
    )
    # Set the seeds for reproducibility
    torch.cuda.manual_seed(config.random_state)
    torch.manual_seed(config.random_state)

    model = AutoModelForCausalLM.from_pretrained(
        config.aligned_model_path,
    )

    # Load the tokenizer and add special tokens
    tokenizer = AutoTokenizer.from_pretrained(config.aligned_model_path)
    tokenizer.pad_token = tokenizer.unk_token

    ################################# load dataset
    dataset = datasets.load_from_disk(dataset_path = config.data_path)
    train_ds = dataset['train']
    val_ds = dataset['test']

    template_func = TEMPLATE_FUNC_MAP[config.data_template]
    og_cols = list(train_ds.column_names)
    train_ds = train_ds.map(template_func, remove_columns=og_cols)
    val_ds = val_ds.map(template_func, remove_columns=og_cols)

    train_dataset = StandardDataset(train_ds, tokenizer, max_words=config.max_seq_length)
    val_dataset = StandardDataset(val_ds, tokenizer, max_words=config.max_seq_length)
    
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.per_device_train_batch_size,
        drop_last=True,
        collate_fn=default_data_collator,
    )
    eval_dataloader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=config.per_device_eval_batch_size,
            drop_last=True,
            collate_fn=default_data_collator,
    )
    # if train_config.use_peft:

    peft_config = LoraConfig(
            r = config.lora_rank,
            lora_alpha = config.lora_alpha, # lora_alpha should match r to maintain scaling = 1
            lora_dropout = config.lora_dropout,
            inference_mode=False,
            init_lora_weights=False, # PiSSA initialization
            target_modules = config.target_modules,
            task_type="CAUSAL_LM",
        )
    
    model = get_peft_model(model, peft_config)
    safelora = SafeLoRA(model, safe_config)
    project_matrix = safelora.project_matrix
    model.print_trainable_parameters()
   
    model.to(safe_config.device)
    

    # Initialize the optimizer and learning rate scheduler
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = StepLR(optimizer, step_size=1, gamma=config.gamma)
    # Start the training process
    results = train(
        model,
        train_dataloader,
        eval_dataloader,
        tokenizer,
        optimizer,
        scheduler,
        config.gradient_accumulation_steps,
        config,
        None,
        None,
        None,
        project_matrix=project_matrix,
        output_dir=save_folder
    )

    [print(f'Key: {k}, Value: {v}') for k, v in results.items()]

if __name__ == "__main__":
    args = parser.parse_args()
    main(args.config_path, args.save_folder)