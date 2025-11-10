import copy, torch, numpy
from transformers import AutoModelForCausalLM
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from peft import PeftModel

from box import Box

from safelora import SafeLoRA
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--config_path', help='Path to config yaml file')
parser.add_argument('--save_folder', help='Folder to save projected model')

'''
Things we need:
1. Base (unaligned) model path
2. (Initial) Aligned model path
3. Finetuned LoRA checkpoint path

What we output:
1. A SafeLoRA finetuned model checkpoint

'''

dtype_map = {"float16": torch.float16,  
                "bfloat16": torch.bfloat16,
                "float32": torch.float32
}

def main(config_path, peft_model_path, save_folder):
    # read config
    config = Box.from_yaml(filename=config_path)
    print("Config:", config)

    os.makedirs(save_folder, exist_ok=True)

    config.to_yaml(os.path.join(save_folder, 'training_config.yaml'))

    peft_path = peft_model_path
    base_model_path = config.base_model_path
    # aligned_model_path = config.aligned_model_path

    #pmodel = AutoModelForCausalLM.from_pretrained(peft_path,dtype=dtype_map[config.dtype])
    base_model = AutoModelForCausalLM.from_pretrained(base_model_path, dtype=dtype_map[config.dtype])
    pmodel = PeftModel.from_pretrained(base_model, peft_path)
    
    safelora = SafeLoRA(pmodel, config)

    proj_model = safelora.model

    # save projected model
    proj_model.save_pretrained(save_folder)




