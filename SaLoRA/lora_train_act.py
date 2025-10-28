#From lora_train_act.py

from safetensors.torch import load_file, save_file
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

import sys
sys.path.append(".")
aligned_model_path = "meta-llama/Meta-Llama-3.1-8B-Instruct"
#Single GPU set-up
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['WORLD_SIZE'] = '1'

#Data preprocessing - function splits batches of examples from dataset into fixed-size blocks of tokens
def group_texts(examples, block_size=128):
    # Concatenate all texts.
    # examples[k] = list of lists, each list contains the tokens of one sample
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]]) #get total number of tokens based on first key ('input_ids')
    total_length = (total_length // block_size) * block_size # length is divisible by block size
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)] # builds list of block-sized chunks
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy() # Next-token prediction - use the correct previous tokens as input
    return result
# Returns a dict like:
#{
# 'input_ids': [[123, 324, ..., 128 (tokens)], [...], ...],
# 'attention_mask': [[1, 1, ...], [...], ...],
# 'labels': [[103, 231, ..., 128 tokens], [...], ...]
#}

from trl import SFTTrainer                          #specialized trainer for supervised fine-tuning of LMs
from datasets import load_dataset, Dataset          #load and manipulate datasets - from HuggingFace
from transformers import Trainer, TrainingArguments #training interface - from HuggingFace
from fastchat.conversation import get_conv_template #formatting conversations with LMs
from tqdm import tqdm                               #"Taqadum" - display progress bars for loops

def parse_args():
    parser = argparse.ArgumentParser(description='Brisque') #built-in Python library to handle CLI arguments
    parser.add_argument('--name', default="output_alphca_chat7b_act12", help='path to input image file')      #output folder name
    parser.add_argument('--rank', type=int, default=16, help='path to input image file')                      #LoRA rank
    parser.add_argument('--rs', type=int, default=32, help='path to input image file')                        #rank for safety subspace
    parser.add_argument('--ds', type=int, default=32, help='path to input image file')                        #rank for task dataset
    parser.add_argument('--bs', type=int, default=16, help='path to input image file')                        #batch size for training
    parser.add_argument('--port', type=int, default=1234, help='path to input image file')                    #master port for distributed training (torch)
    parser.add_argument('--rank_s', type=int, default=32, help='path to input image file')                    #secondary rank param for LoRA
    parser.add_argument('--lr', type=float, default=1e-4, help='path to input image file')                    #learning rate
    parser.add_argument('--wd', type=float, default=0.0, help='path to input image file')                     #weight decay
    parser.add_argument('--more_module', action='store_true', default=False, help='path to input image file') #flag for whether to include additional LoRA modules
    args = parser.parse_args() #parses arguments; each becomes an attribute of args (learning rate = args.lr)
    return args

def main(args_dict=None):
    if args_dict is None:
      args = parse_args()
    else:
      class Args: pass
      args = Args()
      for k, v in args_dict.items():
        setattr(args, k, v)
    os.environ['MASTER_PORT'] = str(args.port)
    #Creates directory path for outputs
    args.name = 'out/'+str(args.rs)+'_'+str(args.ds)+str(args.more_module)+'llama2_alpaca_output/llama3.1_' + args.name +'_rank'+str(args.rank_s)
    try:
        #Creates folder if it does not yet exist
        os.mkdir('out/'+str(args.rs)+'_'+str(args.ds)+str(args.more_module)+'llama3.1_alpaca_output/')
        print("Make dir "+'out/'+str(args.rs)+'_'+str(args.ds)+str(args.more_module)+'llama3.1_alpaca_output/')
    except:
        print("Already Exists")
    #Load pre-trained tokenizer from aligned_model_path
    tokenizer = AutoTokenizer.from_pretrained(aligned_model_path)
    tokenizer.pad_token_id = tokenizer.eos_token_id #LlaMA models don't have a dedicated <pad> token
    model = AutoModelForCausalLM.from_pretrained(aligned_model_path, device_map="cuda:0") #put model on GPU 0
    if args.more_module: #choose to adapt more layers (key, feed-forward up/down projections)
        target_module_list = ["q_proj", "k_proj", "v_proj", "up_proj", "down_proj"]
    else:
        target_module_list = ["q_proj","v_proj"]
    divide_num = len(target_module_list) #store number of target modules

    def tokenize_function(examples):
        return tokenizer(examples["text"])
    adapter_name ='default'
    rank = args.rank
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
    model = make_Act(model, verbose=False) #wraps model so linear layers can record activations
    model.requires_grad_(False)            #freeze base model weights
    model.seqlen = 4096                    #set max sequence length
    clear_act_buffer(model)                #wipe any old stored activations

    #initially, ActLinear layers do not record activates; activated one layer at a time to save memory
    for name, module in model.named_modules():
        if isinstance(module, ActLinear):
            module.record_activation = False
            module.clear_act_buffer()
    #Load data for collecting activations
    dataloader, _ = get_loaders( #regular instruction dataset
        "alpaca_cleaned_no_safety",
        nsamples=128,
        seed=0,
        seqlen=model.seqlen,
        tokenizer=tokenizer,
        disentangle=True,
        modelname='llama2'
    )
    dataloader_safe, _ = get_loaders( #safety-aligned dataset
        'align_short',
        nsamples=128,
        seed=0,
        seqlen=model.seqlen,
        tokenizer=tokenizer,
        disentangle=True,
        modelname='llama2'
    )
    num_hidden_layers = model.config.num_hidden_layers
    weight_list = {}
    current_num = -1
    #At each layer, filter for only modules that belong to that layer
    #For recording activations of one layer at a time
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

        #Run data through model and collect activations
        for batch in dataloader_safe:
            inp, tar = batch[0].to("cuda:0"), batch[1].to("cuda:0")
            assert True, "should run in disentangle mode"
            mask = tar.ne(-100)           #ignore padded tokens
            with set_mask(model, mask):   #ensures only valid tokens are used
                model(inp)                #triggers forward passes --> activations get recorded

        #For each module in current layer: match its name in the target_module_list and concatenate all collected activations into one big tensor
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
                    "cuda:0"
                )  # size * d_in
                score = ( #multiplication producing a matrix representing how activations align with weight directions
                    module.activation_norms @ module.base.weight.data.T
                )  # (size * d_in) @ (d_out * d_in).T --> (size, d_out)
                d_out, d_in = module.base.weight.data.shape
                total_rank = min(d_out, d_in)
                print(
                    f"remaining: rank {name} = {total_rank - rank} / {total_rank}"
                )
                # for removing from the top
                U, S, V = torch.svd_lowrank(
                    score.float(), q=args.rs, niter=n_iter
                )  # (size, r) (r) (d_out, r)
                V =  V.type(module.base.weight.data.dtype) #save V as lora matrix initializer
                #save each V matrix inot a dictionary weight_list, keyed by layer name + index
                #These will be injected into LoRA modules instead of random initialization
                weight_list[layer_n+'_'+str(current_num//divide_num)+'_V'] = V

        #After finishing a layer, stop recording and clear buffers before moving to next layer
        for name, module in model.named_modules():
            if layer_filter_fn(name) and isinstance(module, ActLinear):
                # print("disabling recording for ", name)
                module.record_activation = False
                module.clear_act_buffer()
    
    #filter for only modules belonging to current layer
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
     #runs model forward on data, activations from ActLinear models are stored
        for batch in dataloader:
            inp, tar = batch[0].to("cuda:0"), batch[1].to("cuda:0")
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
                    "cuda:0"
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
                    score.float(), q=args.ds, niter=n_iter
                )  # (size, r) (r) (d_out, r)
                V =  V.type(module.base.weight.data.dtype)
                U2, S2, V2 = torch.svd_lowrank(
                    module.base.weight.data.float(), q=rank, niter=n_iter
                )  # n(size, r) (r) (d_out, r)
                U2 =  U2.type(module.base.weight.data.dtype)
                S2 =  S2.type(module.base.weight.data.dtype)
                V2 =  V2.type(module.base.weight.data.dtype)
                safeV = weight_list[layer_n+'_'+str(current_num//divide_num)+'_V']
                weight_list[layer_n+'_'+str(current_num//divide_num)+'lora_C'] = weight_list[layer_n+'_'+str(current_num//divide_num)+'_V'] @ weight_list[layer_n+'_'+str(current_num//divide_num)+'_V'].T

                #construct LoRA A and B matrices from SVD factors
                #low-rank adapters that approximate effect of full weight update
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
    model = revert_Act_to_Linear(model)         #restores model from ActLinear to standard nn.Linear 
    model.zero_grad()                          
    model = get_peft_model(model, peft_config)  #wraps model with LoRA hooks so only low-rank adapters are trainable
    model.print_trainable_parameters()          #sanity check; confirm only LoRA params are active
    print(model.parameters())

    #replace random initialization of A and B matrices with the computed ones that are safe aligned
    #also attaches lora_C
    current_num = 0
    print(weight_list.keys())
    for name, module in model.named_modules():
        for t_name  in target_module_list:
            if t_name in name:
                if 'lora_A.default' in name:
                    # print(name)
                    module.weight.data = weight_list[t_name+'_'+str(current_num//(2*divide_num))+'lora_A']
                    current_num +=1
                if 'lora_B.default' in name:
                    module.weight.data = weight_list[t_name+'_'+str(current_num//(2*divide_num))+'lora_B']
                    current_num +=1

                if t_name+'.' not in name:
                    # print(name)
                    module.lora_C = weight_list[t_name+'_'+str(current_num//4)+'lora_C'].clone()
                    weight_list[t_name+'_'+str(current_num//4)+'lora_C'] = module.lora_C.clone()
                    module.requires_grad = False #ensures C stays frozen during fine-tuning
                break
    weight_list['divide_num'] = divide_num

    #saves A, B, and C matrices so they can be reloaded later
    torch.save(weight_list, args.name+"_"+str(rank)+"_"+str(args.lr)+'_'+str(args.wd)+'lora_ABC.pt')
    
    del weight_list
    torch.cuda.empty_cache()
    prompts = []
    dataset = load_dataset('yahma/alpaca-cleaned', split="train")
    count = 0
    for line in tqdm(dataset): #reformat into instruction input / output style
        count += 1
        prompts.append({'text':f"[INST]{line['instruction'].strip()} {line['input']}[/INST]{line['output']}"})

    df = Dataset.from_list(prompts)
    training_args = TrainingArguments(
        args.name+"_"+str(rank)+"_"+str(args.lr)+'_'+str(args.wd)+"/",
        evaluation_strategy = "no",
        learning_rate=args.lr,
        weight_decay=args.wd,
        save_steps = 500,
        per_device_train_batch_size=args.bs,
        num_train_epochs=1,
        save_only_model = True,
        report_to = "none",
    )


    tokenized_datasets = df.map(tokenize_function, batched=True, num_proc=4, remove_columns=["text"])
    lm_datasets = tokenized_datasets.map(
    group_texts,
    batched=True,
    batch_size=1024,
    num_proc=4,
    )

    trainer = Trainer( #fine-tuning, only LoRA parameters (A,B) are updated; C is frozen
    model=model,
    args=training_args,
    train_dataset=lm_datasets,
    )
    trainer.train()

if __name__ == '__main__':
    main()
