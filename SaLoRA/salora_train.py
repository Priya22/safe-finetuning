import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
from lowrank_prune.lib.model_wrapper_low import *
from lowrank_prune.lib.data import get_loaders
from peft import LoraConfig, get_peft_model
import time

from trl import SFTTrainer
from datasets import load_dataset, Dataset
from transformers import Trainer, TrainingArguments
from fastchat.conversation import get_conv_template
from tqdm import tqdm

from process_lora import process_checkpoint, update_qv

import logging
import datasets
import csv

from box import Box

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--config_path', help='Path to config file')
parser.add_argument('--save_folder', help='Folder to save finetuned model')

'''
Goal: save SaLoRA-tuned checkpoints at each epoch that can be used for inference

Things we need:
1. Save lora_ABC weights for each layer
2. Save LoRA checkpoints for each epoch
3. Process each checkpoint to merge A and B weights
4. Save back the updated checkpoint

'''

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

def process_checkpoints(save_folder):
    abc_path = os.path.join(save_folder, 'lora_ABC.pt')
    
    weight_list = torch.load(abc_path, map_location=torch.device('cpu'))
    print("Load weight list finished")

    checkpoints = []
    for cf in os.path.scandir(save_folder):
        if "checkpoint-" in cf.name:
            print("processing {}".format(cf.path))
            process_checkpoint(cf.path, weight_list)

            # other step in eval
            update_qv(cf.path, weight_list)

            print("Done.")


def main(config_path, save_folder):
    config = Box.from_yaml(filename=config_path)

    # os.environ['MASTER_PORT'] = str(args.port)

    os.makedirs(save_folder, exist_ok=True)

    # save config to save folder

    config.to_yaml(os.path.join(save_folder, 'config.yaml'))

    # args.name = 'out/'+str(args.rs)+'_'+str(args.ds)+str(args.more_module)+'llama3.1_alpaca_output/llama3.1_' + args.name +'_rank'+str(args.rank_s)
    # try:
    #     os.mkdir('out/'+str(args.rs)+'_'+str(args.ds)+str(args.more_module)+'llama3.1_alpaca_output/')
    #     print("Make dir "+'out/'+str(args.rs)+'_'+str(args.ds)+str(args.more_module)+'llama3.1_alpaca_output/')
    # except:
    #     print("Already Exists")
    aligned_model_path = config.base_model_path

    device = config.device

    tokenizer = AutoTokenizer.from_pretrained(aligned_model_path)
    # tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(aligned_model_path, device=device, dtype=dtype_map[config.dtype], attn_implementation=config.attn_implementation)

    target_module_list = config.target_modules

    divide_num = len(target_module_list)

    adapter_name ='default'

    rank = config.rank
    n_iter=30
    
    peft_config = LoraConfig(
        r = rank,
        lora_alpha = rank, # lora_alpha should match r to maintain scaling = 1
        lora_dropout = 0,
        inference_mode=False,
        init_lora_weights=False, # PiSSA initialization
        target_modules = target_module_list,
        task_type="CAUSAL_LM",
    )
    
    t1 = time.time()
    model = make_Act(model, verbose=False)
    model.requires_grad_(False)
    model.seqlen = 4096
    clear_act_buffer(model)
    
    for name, module in model.named_modules():
        if isinstance(module, ActLinear):
            module.record_activation = False
            module.clear_act_buffer()

    dataloader, _ = get_loaders(
        "alpaca_train",
        nsamples=128,
        seed=0,
        seqlen=model.seqlen,
        tokenizer=tokenizer,
        disentangle=True,
    )
    dataloader_safe, _ = get_loaders(
        'align_short',
        nsamples=128,
        seed=0,
        seqlen=model.seqlen,
        tokenizer=tokenizer,
        disentangle=True,
    )
    num_hidden_layers = model.config.num_hidden_layers
    weight_list = {}
    current_num = -1
    for layer in range(num_hidden_layers):
        layer_filter_fn = (
            lambda x: f"layers.{layer}." in x
        )  ### TODO # hack for llama series

        # enable recording for the current layer.
        for name, module in model.named_modules():
            if layer_filter_fn(name) and isinstance(module, ActLinear):
                # print("enabling recording for ", name)
                # if 'v_proj' in name or 'q_proj' in name:
                module.record_activation = True
                module.clear_act_buffer()
            
        for batch in dataloader_safe:
            inp, tar = batch[0].to(device), batch[1].to(device)
            assert True, "should run in disentangle mode"
            mask = tar.ne(-100)
            with set_mask(model, mask):
                model(inp)

        for name, module in model.named_modules():
            if layer_filter_fn(name) and isinstance(module, ActLinear):
                print("Module name:", name)
                # print(module.activation_norms)
                f_name = ""
                for t_name  in target_module_list:
                    if t_name in name:
                        f_name = t_name
                        break
                if len(f_name)==0:
                    print("Wrong name ", name)
                    continue
                current_num += 1
                layer_n =f_name
                module.activation_norms = torch.cat(module.activation_norms, dim=0).to(
                    device
                )  # size * d_in
                score = (
                    module.activation_norms @ module.base.weight.data.T
                )  # (size * d_in) @ (d_out * d_in).T --> (size, d_out)
                d_out, d_in = module.base.weight.data.shape
                total_rank = min(d_out, d_in)
                print(
                    f"remaining: rank {name} = {total_rank - rank} / {total_rank}"
                )
                # for removing from the top
                U, S, V = torch.svd_lowrank(
                    score.float(), q=config.rs, niter=n_iter
                )  # (size, r) (r) (d_out, r)
                V =  V.type(module.base.weight.data.dtype)

                weight_list[layer_n+'_'+str(current_num//divide_num)+'_V'] = V


        for name, module in model.named_modules():
            if layer_filter_fn(name) and isinstance(module, ActLinear):
                # print("disabling recording for ", name)
                module.record_activation = False
                module.clear_act_buffer()
    current_num = -1
    for layer in range(num_hidden_layers):
        layer_filter_fn = (
            lambda x: f"layers.{layer}." in x
        )  ### TODO # hack for llama series

        # enable recording for the current layer.
        for name, module in model.named_modules():
            if layer_filter_fn(name) and isinstance(module, ActLinear):
                # print("enabling recording for ", name)
                # if 'v_proj' in name or 'q_proj' in name:
                module.record_activation = True
            
        # forward pass and get activation records.

        # with torch.no_grad():
        for batch in dataloader:
            inp, tar = batch[0].to(device), batch[1].to(device)
            assert True, "should run in disentangle mode"
            mask = tar.ne(-100)
            with set_mask(model, mask):
                model(inp)

        for name, module in model.named_modules():
            if layer_filter_fn(name) and isinstance(module, ActLinear):
                # print("Module name:", name)
                # print(module.activation_norms)
                f_name = ""
                for t_name  in target_module_list:
                    if t_name in name:
                        f_name = t_name
                        break
                if len(f_name)==0:
                    # print("Wrong name ", name)
                    continue
                current_num += 1
                layer_n = f_name
                module.activation_norms = torch.cat(module.activation_norms, dim=0).to(
                    device
                )  # size * d_in
                score = (
                    module.activation_norms @ module.base.weight.data.T
                )  # (size * d_in) @ (d_out * d_in).T --> (size, d_out)
                d_out, d_in = module.base.weight.data.shape
                total_rank = min(d_out, d_in)
                print(
                    f"remaining: rank {name} = {total_rank - rank} / {total_rank}"
                )

                # for removing from the bottom
                U, S, V = torch.svd_lowrank(
                    score.float(), q=config.ds, niter=n_iter
                )  # (size, r) (r) (d_out, r)
                V =  V.type(module.base.weight.data.dtype)

                U2, S2, V2 = torch.svd_lowrank(
                    module.base.weight.data.float(), q=rank, niter=n_iter
                )  # n(size, r) (r) (d_out, r)
                U2 =  U2.type(module.base.weight.data.dtype)
                S2 =  S2.type(module.base.weight.data.dtype)
                V2 =  V2.type(module.base.weight.data.dtype)

                safeV = weight_list[layer_n+'_'+str(current_num//divide_num)+'_V']
                weight_list[layer_n+'_'+str(current_num//divide_num)+'lora_C'] = torch.eye(safeV.size(0),dtype=module.base.weight.dtype, device=module.base.weight.device) - safeV @ safeV.T
            
                weight_list[layer_n+'_'+str(current_num//divide_num)+'lora_B'] = U2 @ torch.diag(torch.sqrt(S2))
                weight_list[layer_n+'_'+str(current_num//divide_num)+'lora_B'] = V @ V.T @ weight_list[layer_n+'_'+str(current_num//divide_num)+'lora_B']
                weight_list[layer_n+'_'+str(current_num//divide_num)+'lora_A'] = torch.diag(torch.sqrt(S2)) @ V2.T
                module.base.weight.data.sub_(
                    weight_list[layer_n+'_'+str(current_num//divide_num)+'lora_C'] @ weight_list[layer_n+'_'+str(current_num//divide_num)+'lora_B'] @ weight_list[layer_n+'_'+str(current_num//divide_num)+'lora_A'])
                weight_list[layer_n+'_'+str(current_num//divide_num)+'weight'] = module.base.weight.data
                # print(weight_list[layer_n+'_'+str(current_num//divide_num)+'lora_B'].size(), weight_list[layer_n+'_'+str(current_num//divide_num)+'lora_A'].size())
            # torch.cuda.empty_cache()

    # disable recording for the current layer.
        for name, module in model.named_modules():
            if layer_filter_fn(name) and isinstance(module, ActLinear):
                # print("disabling recording for ", name)
                module.record_activation = False
                module.clear_act_buffer()
    t2 = time.time()
    print("Time cost", t2-t1)
    model = revert_Act_to_Linear(model)
    model.zero_grad()
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    # print(model.parameters())

    print(weight_list.keys())

    module_counters = {t_name: 0 for t_name in target_module_list}

    for name, module in model.named_modules():
        for t_name in target_module_list:
            if t_name in name:
                layer_idx = module_counters[t_name] // divide_num

                if 'lora_A.default' in name:
                    key = f'{t_name}_{layer_idx}lora_A'
                    if key in weight_list:
                        module.weight.data = weight_list[key]
                        print(f"Set {name} with {key}")
                    else:
                        print(f"Warning: {key} not found")

                elif 'lora_B.default' in name:
                    key = f'{t_name}_{layer_idx}lora_B'
                    if key in weight_list:
                        module.weight.data = weight_list[key]
                        print(f"Set {name} with {key}")
                    else:
                        print(f"Warning: {key} not found")

                elif t_name + '.' not in name:  # Main module (not lora_A or lora_B)
                    key = f'{t_name}_{layer_idx}lora_C'
                    if key in weight_list:
                        module.lora_C = weight_list[key].clone()
                        weight_list[key] = module.lora_C.clone()
                        module.requires_grad = False
                        print(f"Set {name} lora_C with {key}")
                    else:
                        print(f"Warning: {key} not found")

                    module_counters[t_name] += 1

                break
    weight_list['divide_num'] = divide_num
    
    # save_path = args.name+'_'+str(rank) + '_' + str(args.lr) + '_' + str(args.wd) + 'lora_ABC.pt'
    save_path = os.path.join(save_folder, 'lora_ABC.pt')
    # os.makedirs(os.path.dirname(save_path), exist_ok = True)    
    torch.save(weight_list, save_path)
    del weight_list
    torch.cuda.empty_cache()

    # LoRA training
    prompts = []
    logging.info("loading dataset")

    data_path = config.data_path
    dataset = datasets.load_from_disk(data_path)
    train_ds = dataset['train']
    val_ds = dataset['test']

    template_func = TEMPLATE_FUNC_MAP[config.data_template]

    og_cols = list(train_ds.column_names)
    train_ds = train_ds.map(template_func, remove_columns=og_cols)
    val_ds = val_ds.map(template_func, remove_columns=og_cols)

    # rows = []
    # for i in range(len(train_ds)):
    #     sample = train_ds[i]
    #     input = sample['instruction'] + ': ' + sample['input']
    #     output = sample['output']
    #     rows.append([i, input, output])

    # header = ['index', 'prompt', 'response']
    # with open("/content/drive/MyDrive/CAISI_phase2/SaLoRA-finetuning/SaLoRA-Repo/lowrank_prune/data/alpaca_5k_train.csv", "w") as f:
    #     writer = csv.writer(f)
    #     writer.writerow(header)
    #     for row in rows:
    #         writer.writerow(row)

    # df = Dataset.from_list(prompts)
    training_args = TrainingArguments(
        # args.name+"_"+str(rank)+"_"+str(args.lr)+'_'+str(args.wd)+"/",
        learning_rate=config.lr,
        lr_scheduler_type=config.lr_scheduler_type,
        warmup_ratio=config.warmup_ratiom
        weight_decay=config.wd,
        optim=config.optim,
        gradient_checkpointing=config.use_gradient_checkpointing,
        per_device_train_batch_size=config.bs,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        max_length=config.max_seq_length,
        packing=config.use_packing,
        num_train_epochs=config.num_epochs,
        save_only_model = False,
        eval_strategy = config.eval_strategy,
        save_strategy = config.save_strategy,
        logging_first_step=True,
        logging_strategy=config.logging_strategy,
        logging_steps=config.logging_steps,
        logging_dir=save_folder,
        log_level="info",
        seed=config.random_state,
        report_to = "none",
        output_dir=save_folder,
        eos_token=tokenizer.eos_token
    )
    # user_prompt = traindata_sampled["prompt"][i]
    # assistant_response = traindata_sampled["response"][i]
    # formatted_messages = [
    #     {"role": "user", "content": user_prompt},
    #     {"role": "assistant", "content": assistant_response},
    # ]
    
    # tokenized_datasets = df.map(tokenize_function, batched=True, num_proc=4, remove_columns=["text"])
    # lm_datasets = tokenized_datasets.map(
    # group_texts,
    # batched=True,
    # batch_size=1024,
    # num_proc=4,
    # )

    trainer = Trainer(
    model=model, # peft model
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    processing_class=tokenizer
    )

    trainer.train()

    #save final model
    tokenizer.save_pretrained(save_folder)
    trainer.save_model(output_dir = save_folder)

    ## further processing of weights



if __name__ == '__main__':
    args = parser.parse_args()
    config_path = args.config_path
    save_folder = args.save_folder
    main(config_path, save_folder)

    #post-processing
    print("Running post-processing steps for checkpoints")
    process_checkpoints(save_folder)

