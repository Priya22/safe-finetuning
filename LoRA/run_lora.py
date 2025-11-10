import os, re, sys, json, string, csv, gzip
import pandas as pd
import numpy as np

from box import Box

import datasets
import torch
import peft
import trl
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset, load_from_disk

from peft import get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from trl import apply_chat_template
import argparse
import logging
import time

os.environ["WANDB_DISABLED"] = "true"

parser = argparse.ArgumentParser()
parser.add_argument('--config_path', help='path to config file', required=True)
parser.add_argument('--save_folder', help='path to save outputs and log files')


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

class LoraFT:
    def __init__(self, config, save_folder):
        self.config = config
        self.save_folder = save_folder

        self.model, self.tokenizer = self.load_model_for_training()

        self.train_ds, self.val_ds = self.load_dataset_for_training()


    def load_model_for_training(self):
        config = self.config
        logging.info("Loading base model and tokenizer")
        

        model = AutoModelForCausalLM.from_pretrained(config.base_model_path,
               dtype=dtype_map[config.dtype]) #, attn_implementation=config.attn_implementation)
        # model = prepare_model_for_kbit_training(model)

        tokenizer = AutoTokenizer.from_pretrained(config.base_model_path)
        # tokenizer.pad_token = tokenizer.eos_token

        return model, tokenizer
    
    def load_dataset_for_training(self):
        logging.info("loading dataset")
        config = self.config

        # data_path = config.data_path
        data_format = config.data_template

        train_ds = load_from_disk(os.path.join(config.data_path, "train"))
        val_ds = load_from_disk(os.path.join(config.data_path, "test"))
        logging.info("Dataset info: {} ".format(train_ds.info))
        og_cols = list(train_ds.column_names)

        template_func = TEMPLATE_FUNC_MAP[data_format] # add others later


        logging.info("Splits train and val with {} and {} instances".format(len(train_ds), len(val_ds)))

        logging.info("Standardizing to prompt-completion format for TRL")

        # for k in dataset:
        train_ds = train_ds.map(template_func, remove_columns=og_cols)
        val_ds = val_ds.map(template_func, remove_columns=og_cols)

        return train_ds, val_ds

    def run_peft_training(self):
        logging.info("Preparing PEFT model")
        config = self.config
        peft_config = LoraConfig(
                r = config.lora_rank,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                bias="none",
                target_modules=config.target_modules,
                task_type='CAUSAL_LM'
            )
        
        # model = get_peft_model(self.model, peft_config)
        train_ds, val_ds = self.train_ds, self.val_ds
        tokenizer = self.tokenizer
        model = self.model

        # a,b = model.get_nb_trainable_parameters()
        # logging.info("{} trainable parameters | ratio to overall: {}".format(a, a/b))

        training_args = SFTConfig(
                # dataset_text_field='messages',
                learning_rate=config.learning_rate,
                optim=config.optim,
                use_liger_kernel=config.use_liger_kernel,
                gradient_checkpointing=config.use_gradient_checkpointing,
                num_train_epochs=config.num_train_epochs,
                per_device_train_batch_size=config.per_device_train_batch_size,
                per_device_eval_batch_size=config.per_device_train_eval_size,
                gradient_accumulation_steps=config.gradient_accumulation_steps,
                warmup_ratio=config.warmup_ratio,
                lr_scheduler_type=config.lr_scheduler_type,
                max_grad_norm=config.max_grad_norm,
                weight_decay = config.weight_decay,
                max_length=config.max_seq_length,
                packing=config.use_packing,
                output_dir=self.save_folder,
                save_strategy=config.save_strategy,
                save_only_model=config.save_only_model,
                eval_strategy=config.eval_strategy,
                eval_steps=config.eval_steps if config.eval_strategy=='steps' else None,
                logging_dir=self.save_folder,
                logging_first_step=True,
                logging_strategy=config.logging_strategy,
                logging_steps=config.logging_steps,
                log_level='info',
                seed=config.random_state,
                eos_token=tokenizer.eos_token
            )
        
        trainer = SFTTrainer(
                model = model,
                # tokenizer=tokenizer,
                # data_collator = DataCollatorForSeq2Seq(tokenizer = tokenizer),
                args=training_args,
                peft_config=peft_config,
                train_dataset=train_ds,
                eval_dataset=val_ds,
                # formatting_func=formatting_prompts_func
                processing_class=tokenizer,
            )
        
        trainer.accelerator.print(f"{trainer.model}")
        if hasattr(trainer.model, "print_trainable_parameters"):
            trainer.model.print_trainable_parameters()
        
        trainer.train()

        # if trainer.is_fsdp_enabled:
        #     trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
        # trainer.model.config.use_cache = True
        # save final adapter
        # trainer.save_model(output_dir = self.save_folder)

        # training_args.distributed_state.wait_for_everyone()
        tokenizer.save_pretrained(training_args.output_dir)
        logging.info(f'Tokenizer saved to {training_args.output_dir}')


        logging.info("Finished training | final model saved.")


def main(config_path, save_folder):
    config = Box.from_yaml(filename=config_path)

    
    
    logging.info("Read config from {}".format(config_path))
    logging.info("Config: {}".format(config))

    exp_id = config.exp_id
    save_folder = os.path.join(save_folder, exp_id)
    os.makedirs(save_folder, exist_ok=True)


    # logging.info("random state: {}".format(random_state))
    config.to_yaml(filename=os.path.join(save_folder, "training_config.yaml"))
    logging.info("Init Model and Dataset..")

    tuner = LoraFT(config=config, save_folder=save_folder)

    logging.info("Starting training at {}".format(time.time()))

    tuner.run_peft_training()

    logging.info("Job done at {}".format(time.time()))


if __name__ == '__main__':
    args = parser.parse_args()
    save_folder = args.save_folder
    os.makedirs(save_folder, exist_ok=True)

    # log_file = os.path.join(save_folder, 'log.txt') # goes to slurm log
    logging.basicConfig(encoding='utf-8', level=logging.DEBUG, \
                        format='%(asctime)s %(message)s')
    
    main(args.config_path, save_folder)