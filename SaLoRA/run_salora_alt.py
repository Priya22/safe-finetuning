import os, re, csv, gc, time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from safetensors.torch import load_file, save_file
import datasets


from lowrank_prune.lib.model_wrapper_low import *
# from lowrank_prune.lib.data import get_loaders
from peft import LoraConfig, get_peft_model

from trl import SFTTrainer, SFTConfig
from datasets import load_dataset, Dataset
from fastchat.conversation import get_conv_template
from tqdm import tqdm

# from process_lora import process_checkpoint
import logging


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

def process_checkpoint(checkpoint_folder, weight_list, num_layers=32):

    a = load_file(os.path.join(checkpoint_folder, 'adapter_model.safetensors'),device='cpu')
    print("Loaded adapter safetensors")

    # change name
    os.rename(os.path.join(checkpoint_folder, 'adapter_model.safetensors'), os.path.join(checkpoint_folder, 'adapter_model_ori.safetensors'))

    print("num layers: ", num_layers)

    for idx in range(num_layers):

        a['base_model.model.model.layers.'+str(idx)+'.self_attn.q_proj.lora_B.weight'] = weight_list['q_proj_'+str(idx)+'lora_C'] @ a['base_model.model.model.layers.'+str(idx)+'.self_attn.q_proj.lora_B.weight']
        a['base_model.model.model.layers.'+str(idx)+'.self_attn.v_proj.lora_B.weight'] = weight_list['v_proj_'+str(idx)+'lora_C'] @ a['base_model.model.model.layers.'+str(idx)+'.self_attn.v_proj.lora_B.weight']

    print("update finished. Saving to {}/adapter_model.safetensors".format(checkpoint_folder))

    save_file(a, os.path.join(checkpoint_folder, 'adapter_model.safetensors'))

def main(config_path, save_folder):

    config = Box.from_yaml(filename=config_path)
    exp_id = config.exp_id

    save_folder = os.path.join("..", "outputs", config.exp_id)
    os.makedirs(save_folder, exist_ok=True)

    print("saving all artifacts to {}".format(save_folder))
    # save config to save folder

    config.to_yaml(os.path.join(save_folder, 'training_config.yaml'))

    print(config)

    model_path = config.base_model_path

    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", dtype=dtype_map[config.dtype])
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    device = model.device


    dataloader = get_loaders(
            "alpaca_train",
            nsamples=500,
            seed=0,
            seqlen=4096,
            tokenizer=tokenizer,
            disentangle=True,
        )


    dataloader_safe = get_loaders(
            'align_train',
            nsamples=None,
            seed=0,
            seqlen=4096,
            tokenizer=tokenizer,
            disentangle=True,
        )


    model = make_Act(model, verbose=False)

    model.requires_grad_(False)

    clear_act_buffer(model)


    for name, module in model.named_modules():
        if isinstance(module, ActLinear):
            # print(name)
            module.record_activation = False
            module.clear_act_buffer()

    target_module_list = config.target_modules
    divide_num = len(target_module_list)
    n_iter = 30

    rank = config.lora_rank

    num_hidden_layers = model.config.num_hidden_layers

    weight_list = {}
    current_num = -1
    for layer in range(num_hidden_layers):
        print(f"layer {layer}")
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
                    # print("Wrong name ", name)
                    continue
                current_num += 1
                layer_n =f_name
                print(layer_n)
                module.activation_norms = torch.cat(module.activation_norms, dim=0).to(
                    device
                )  # size * d_in
                score = (
                    module.activation_norms @ module.base.weight.data.T
                )  # (size * d_in) @ (d_out * d_in).T --> (size, d_out)
                d_out, d_in = module.base.weight.data.shape
                # print(d_out, d_in)
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

        print(f"layer {layer}")
        layer_filter_fn = (
            lambda x: f"layers.{layer}." in x
        )  ### TODO # hack for llama series

        # enable recording for the current layer.
        for name, module in model.named_modules():
            if layer_filter_fn(name) and isinstance(module, ActLinear):
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
                print(layer_n)
                module.activation_norms = torch.cat(module.activation_norms, dim=0).to(
                    device
                )  # size * d_in
                score = (
                    module.activation_norms @ module.base.weight.data.T
                )  # (size * d_in) @ (d_out * d_in).T --> (size, d_out)
                d_out, d_in = module.base.weight.data.shape
                # print(d_out, d_in)
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

                ### don't fix the mistake
                safeV = weight_list[layer_n+'_'+str(current_num//divide_num)+'_V']
                weight_list[layer_n+'_'+str(current_num//divide_num)+'lora_C'] = safeV @ safeV.T

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


    model = revert_Act_to_Linear(model)
    model.zero_grad()

    peft_config = LoraConfig(
            r = rank,
            lora_alpha = config.lora_alpha, # lora_alpha should match r to maintain scaling = 1
            lora_dropout = config.lora_dropout,
            inference_mode=False,
            init_lora_weights=False, # PiSSA initialization
            target_modules = target_module_list,
            task_type="CAUSAL_LM",
        )

    model = get_peft_model(model, peft_config)

    model.print_trainable_parameters()

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

    save_path = os.path.join(save_folder, 'lora_ABC.pt')

    torch.save(weight_list, save_path)

    del weight_list
    # del old_weights
    del dataloader
    del dataloader_safe

    gc.collect()

    torch.cuda.empty_cache()

    data_path = config.data_path
    dataset = datasets.load_from_disk(data_path)
    train_ds = dataset['train']
    val_ds = dataset['test']

    template_func = TEMPLATE_FUNC_MAP[config.data_template]
    og_cols = list(train_ds.column_names)
    train_ds = train_ds.map(template_func, remove_columns=og_cols)
    val_ds = val_ds.map(template_func, remove_columns=og_cols)

    model.get_nb_trainable_parameters()

    training_args = SFTConfig(
            learning_rate=config.learning_rate,
            lr_scheduler_type=config.lr_scheduler_type,
            use_liger_kernel=config.use_liger_kernel,
            warmup_ratio=config.warmup_ratio,
            weight_decay=config.weight_decay,
            # max_grad_norm=config.max_grad_norm,
            optim=config.optim,
            gradient_checkpointing=config.use_gradient_checkpointing,
            per_device_train_batch_size=config.per_device_train_batch_size,
            per_device_eval_batch_size=config.per_device_eval_batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            max_length=config.max_seq_length,
            packing=config.use_packing,
            num_train_epochs=config.num_train_epochs,
            save_only_model = config.save_only_model,
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

    trainer = SFTTrainer(
            model=model, # peft model
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            processing_class=tokenizer
        )


    trainer.train()

    tokenizer.save_pretrained(save_folder)
    print("Saved to {}".format(save_folder))

    print("Running post-processing on checkpoints: ")

    abc_path = os.path.join(save_folder, 'lora_ABC.pt')
    weight_list = torch.load(abc_path, map_location=torch.device('cpu'))


    checkpoint_folder = save_folder
    for cf in os.scandir(checkpoint_folder):
        if "checkpoint" in cf.name:
            print(cf.path)
            process_checkpoint(cf.path, weight_list, num_layers=num_hidden_layers)


if __name__ == '__main__':
    args = parser.parse_args()
    config_path = args.config_path
    save_folder = args.save_folder
    main(config_path, save_folder)