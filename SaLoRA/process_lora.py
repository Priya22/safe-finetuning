from safetensors.torch import load_file, save_file
import torch
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
from lowrank_prune.lib.model_wrapper_low import *
from lowrank_prune.lib.data import get_loaders
from peft import LoraConfig, get_peft_model

from trl import SFTTrainer
from datasets import load_dataset, Dataset
from transformers import Trainer, TrainingArguments
from fastchat.conversation import get_conv_template
from tqdm import tqdm

# def parse_args():
#     parser = argparse.ArgumentParser(description='Brisque')
#     parser.add_argument('--main_path', default="./out/Falsellama2_commonsense_olora_output/llama2_output_alphca_chat7b_act12_rank32True_16_0.0001_0.0/checkpoint-4208", help='path to input image file')
#     parser.add_argument('--weight_path', default='./out/Falsellama2_commonsense_olora_output/llama2_output_alphca_chat7b_act12_rank32True_16_0.0001_0.0lora_ABC.pt', help='path to input image file')
#     args = parser.parse_args()
#     return args
# def main():
    # args = parse_args()
    # mainpath = args.main_path

def process_checkpoint(checkpoint_folder, weight_list):

    a = load_file(os.path.join(checkpoint_folder, '/adapter_model.safetensors'),device='cpu')
    print("Loaded adapter safetensors")

    # change name
    os.rename(os.path.join(checkpoint_folder, 'adapter_model.safetensors'), os.path.join(checkpoint_folder, 'adapter_model_ori.safetensors'))

    weight_layer_nums = set()
    pattern = r'\d+\.?\d*' # match numeric sequences
    for k in weight_list.keys():
        weight_layer_nums.update([int(x) for x in re.findall(pattern, k)])

    num_layers = max(weight_layer_nums)+1 # TBD: automatically find

    for idx in range(num_layers):

        a['base_model.model.model.layers.'+str(idx)+'.self_attn.q_proj.lora_B.weight'] = weight_list['q_proj_'+str(idx)+'lora_C'] @ a['base_model.model.model.layers.'+str(idx)+'.self_attn.q_proj.lora_B.weight']
        a['base_model.model.model.layers.'+str(idx)+'.self_attn.v_proj.lora_B.weight'] = weight_list['v_proj_'+str(idx)+'lora_C'] @ a['base_model.model.model.layers.'+str(idx)+'.self_attn.v_proj.lora_B.weight']

    print("update finished. Saving to {}/adapter_model.safetensors")

    save_file(a, os.path.join(checkpoint_folder, 'adapter_model.safetensors'))


def update_qv(checkpoint_folder, weight_list):
    current_num = 0

    model = AutoModelForCausalLM.from_pretrained(checkpoint_folder)
    for name, module in model.named_modules():
        if 'self_attn.qkv_proj.base_layer' in name:
            print(module.weight.size(), weight_list['q_proj_'+str(current_num//2)+'weight'].size())
            print(name,'q_proj_'+str(current_num//2)+'weight')

            tmp_data = torch.cat([weight_list['q_proj_'+str(current_num//2)+'weight'].to(module.weight.dtype).to(module.weight.device), 
                                            module.weight.data[weight_list['q_proj_'+str(current_num//2)+'weight'].size(0):-weight_list['v_proj_'+str(current_num//2)+'weight'].size(0),:],
                                            weight_list['v_proj_'+str(current_num//2)+'weight'].to(module.weight.dtype).to(module.weight.device)],dim=0)
            print('final_size',module.weight.size(), tmp_data.size())
            
            module.weight.copy_(tmp_data)
            print(torch.norm(module.weight), torch.norm(tmp_data))
            current_num += 2

    print("Saving model to {}".format(checkpoint_folder))
    model.save_pretrained(checkpoint_folder)

# if __name__ == '__main__':
#     main()

