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

def main(config_path, save_folder):
    # read config
    config = Box.from_yaml(filename=config_path)
    #print("Config:", config)
    os.makedirs(save_folder, exist_ok=True)
    config.to_yaml(os.path.join(save_folder, 'training_config.yaml'))

    peft_path = config.peft_model_path
    base_model_path = config.base_model_path
    # aligned_model_path = config.aligned_model_path

    #pmodel = AutoModelForCausalLM.from_pretrained(peft_path,dtype=dtype_map[config.dtype])
    base_model = AutoModelForCausalLM.from_pretrained(base_model_path, torch_dtype=dtype_map[config.dtype])

    #Iterate through each individual checkpoint
    checkpoints = [d for d in os.listdir(peft_path) if d.startswith("checkpoint")]
    checkpoints.sort(key=lambda x: int(x.split("-")[1]))

    for ckpt in checkpoints:
        ckpt_path = os.path.join(peft_path, ckpt)
        print(f"Processing checkpoint: {ckpt_path}")
        pmodel = PeftModel.from_pretrained(base_model, ckpt_path, torch_dtype=dtype_map[config.dtype])
        
        safelora = SafeLoRA(pmodel, config)

        proj_model = safelora.model
        save_ckpt = os.path.join(save_folder, ckpt)

        # save projected model
        proj_model.save_pretrained(save_ckpt)
        del pmodel
        del safelora
        del proj_model
        torch.cuda.empty_cache()

if __name__ == "__main__":
    args = parser.parse_args()
    main(args.config_path, args.save_folder)



