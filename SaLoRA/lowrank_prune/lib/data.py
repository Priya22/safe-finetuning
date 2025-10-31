# Code adapted from https://github.com/IST-DASLab/sparsegpt/blob/master/datautils.py

import numpy as np
import random
import torch
from datasets import load_dataset, load_from_disk


# Set seed for reproducibility
def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)


# Wrapper for tokenized input IDs
class TokenizerWrapper:
    def __init__(self, input_ids):
        self.input_ids = input_ids


# Load and process aligned dataset
def get_align(nsamples, seed, seqlen, tokenizer, disentangle=False, mode="base"):
    # Load train and test datasets
    # if mode == "short":
    #     data_files = {"train": "lowrank_prune/data/SFT_aligned_llama2-7b-chat-hf_train_short.csv"}
    # elif mode == "short025":
    #     data_files = {"train": "lowrank_prune/data/SFT_aligned_llama2-7b-chat-hf_train_short025.csv"}
    # elif mode == "short05":
    #     data_files = {"train": "lowrank_prune/data/SFT_aligned_llama2-7b-chat-hf_train_short05.csv"}
    # elif mode == "short075":
    #     data_files = {"train": "lowrank_prune/data/SFT_aligned_llama2-7b-chat-hf_train_short075.csv"}
    # elif mode == "mis":
    #     data_files = {"train": "/home/c01mili/CISPA-projects/llm_ftsec-2024/PiSSA/mistral_harm.csv"}
    # else:
    #     data_files = {"train": "lowrank_prune/data/SFT_aligned_llama2-7b-chat-hf_train.csv"}
    data_file = "/content/drive/MyDrive/CAISI_phase2/safe-finetuning/SaLoRA/lowrank_prune/data/align_train.csv"
    traindata = load_dataset("csv", data_files=data_file, split="train")
    trainloader = []
    random.seed(seed)
    # if disentangle:
    traindata_sampled = traindata.shuffle(seed=seed).select(range(nsamples))
    for i in range(nsamples):
        user_prompt = traindata_sampled["input"][i]
        assistant_response = traindata_sampled["output"][i]
        formatted_inp = [{"role": "user", "content": user_prompt}]
        prompt_enc = tokenizer.apply_chat_template(
        formatted_inp,
           tokenize=True,
           return_tensors="pt",
           add_generation_prompt=True
        )
        response_enc = tokenizer(
            assistant_response,
            add_special_tokens=False,
            return_tensors="pt"
        )
        
        inp = torch.cat((prompt_enc, response_enc["input_ids"]), dim=1)
        
        prompt_len = prompt_enc.shape[1]
        
        tar = inp.clone()
        tar[:, :prompt_len] = -100
        trainloader.append((inp, tar))
        
    # else:
    #     # Encode datasets
    #     trainenc = tokenizer(" ".join(traindata["text"]), return_tensors="pt")

    #     # Generate samples from training set
    #     for _ in range(nsamples):
    #         i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
    #         j = i + seqlen
    #         inp = trainenc.input_ids[:, i:j]
    #         tar = inp.clone()
    #         tar[:, :-1] = -100
    #         trainloader.append((inp, tar))
    return trainloader


# Load and process wikitext2 dataset
def get_wikitext2(nsamples, seed, seqlen, tokenizer):
    # Load train and test datasets
    testdata = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    testenc = tokenizer("\n\n".join(testdata["text"]), return_tensors="pt")

    return None, testenc


def get_alpaca(nsamples, seed, seqlen, tokenizer, disentangle=False, dataset="alpaca_train"):
    data_files = {"no_safety_train": "lowrank_prune/data/alpaca_cleaned_no_safety_train.csv", 
                  "alpaca_train": "/content/drive/MyDrive/CAISI_phase2/safe-finetuning/ft_datasets/alpaca_clean_sample_5k"}

    # traindata = load_dataset("csv", data_files=data_files, split="train")
    traindata = load_from_disk(data_files[dataset])
    if isinstance(traindata, dict) or "train" in traindata:
        traindata = traindata["train"]
    random.seed(seed)
    
    # Encode datasets
    trainloader = []
    # if disentangle:
    traindata_sampled = traindata.shuffle(seed=seed).select(range(nsamples))
    for i in range(nsamples):

        prompt = traindata_sampled["instruction"][i] + ": " + traindata_sampled["input"][i]
        response = traindata_sampled["output"][i]

        prompt_enc = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=True, return_tensors="pt")

        response_enc = tokenizer(response, add_special_tokens=False, return_tensors="pt")

        inp = torch.cat(
            (prompt_enc, response_enc.input_ids), dim=1
        )

        prompt_len = prompt_enc.shape[1]

        tar = inp.clone()
        tar[:, :prompt_len] = -100
        trainloader.append((inp, tar))
    # else:
    #     trainenc = tokenizer(" ".join(traindata["text"]), return_tensors="pt")
    #     # Generate samples from training set
    #     for _ in range(nsamples):
    #         i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
    #         j = i + seqlen
    #         inp = trainenc.input_ids[:, i:j]
    #         tar = inp.clone()
    #         tar[:, :-1] = -100
    #         trainloader.append((inp, tar))
    return trainloader
    
def get_commonsense(nsamples, seed, seqlen, tokenizer, disentangle=False, dataset="commonsense_short"):
    if dataset == 'commonsense_short':
        data_files = {"train": "/home/c01mili/CISPA-projects/llm_ftsec-2024/PiSSA/commonsense_17k.csv"}
    else:
        data_files = {"train": "/home/c01mili/CISPA-projects/llm_ftsec-2024/PiSSA/commonsense_170k.csv"}
    traindata = load_dataset("csv", data_files=data_files, split="train")
    random.seed(seed)
    # Encode datasets
    trainloader = []
    if disentangle:
        traindata_sampled = traindata.shuffle(seed=seed).select(range(nsamples))
        for i in range(nsamples):
            trainenc_prompt = tokenizer(
                traindata_sampled["prompt"][i], return_tensors="pt"
            )
            trainenc_response = tokenizer(
                traindata_sampled["response"][i], return_tensors="pt"
            )
            inp = torch.cat(
                (trainenc_prompt.input_ids, trainenc_response.input_ids[:, 1:]), dim=1
            )  # to remove the first token of the response ('1')
            tar = inp.clone()
            trainenc_prompt_len = trainenc_prompt.input_ids.shape[1]
            tar[:, :trainenc_prompt_len] = -100
            trainloader.append((inp, tar))
    else:
        trainenc = tokenizer(" ".join(traindata["text"]), return_tensors="pt")
        # Generate samples from training set
        for _ in range(nsamples):
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = trainenc.input_ids[:, i:j]
            tar = inp.clone()
            tar[:, :-1] = -100
            trainloader.append((inp, tar))
    return trainloader, None

def get_med(nsamples, seed, seqlen, tokenizer, disentangle=False, dataset="med", model='mistral'):
    data_files = {"train": "/home/c01mili/CISPA-projects/llm_ftsec-2024/PiSSA/doctor_50k.csv"}
    traindata = load_dataset("csv", data_files=data_files, split="train")
    random.seed(seed)
    # Encode datasets
    trainloader = []
    if disentangle:
        traindata_sampled = traindata.shuffle(seed=seed).select(range(nsamples))
        for i in range(nsamples):
            trainenc_prompt = tokenizer(
                traindata_sampled["prompt"][i], return_tensors="pt"
            )
            trainenc_response = tokenizer(
                traindata_sampled["response"][i], return_tensors="pt"
            )
            inp = torch.cat(
                (trainenc_prompt.input_ids, trainenc_response.input_ids[:, 1:]), dim=1
            )  # to remove the first token of the response ('1')
            tar = inp.clone()
            trainenc_prompt_len = trainenc_prompt.input_ids.shape[1]
            tar[:, :trainenc_prompt_len] = -100
            trainloader.append((inp, tar))
    else:
        trainenc = tokenizer(" ".join(traindata["text"]), return_tensors="pt")
        # Generate samples from training set
        for _ in range(nsamples):
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = trainenc.input_ids[:, i:j]
            tar = inp.clone()
            tar[:, :-1] = -100
            trainloader.append((inp, tar))
    return trainloader, None


# Function to select the appropriate loader based on dataset name
def get_loaders(
    name, nsamples=128, seed=0, seqlen=2048, tokenizer=None, disentangle=False
):
    if name == "wikitext":
        return get_wikitext2(nsamples, seed, seqlen, tokenizer)
    if name in ["alpaca", "alpaca_cleaned", "alpaca_cleaned_no_safety", "alpaca_train"]:
        return get_alpaca(nsamples, seed, seqlen, tokenizer, disentangle, dataset=name)
    if name == "align":
        return get_align(nsamples, seed, seqlen, tokenizer, disentangle=disentangle)
    if name == "align_short":
        return get_align(
            nsamples, seed, seqlen, tokenizer, disentangle=disentangle, mode="short")
    if name == "align_short025":
        return get_align(
            nsamples, seed, seqlen, tokenizer, disentangle=disentangle, mode="short025")
    if name == "align_short05":
        return get_align(
            nsamples, seed, seqlen, tokenizer, disentangle=disentangle, mode="short05")
    if name == "align_short075":
        return get_align(
            nsamples, seed, seqlen, tokenizer, disentangle=disentangle, mode="short075")
    if name == "align_short_mis":
        return get_align(
            nsamples, seed, seqlen, tokenizer, disentangle=disentangle, mode="mis")
    if name in ["commonsense","commonsense_short"]:
        return get_commonsense(nsamples, seed, seqlen, tokenizer, disentangle=disentangle, dataset=name)
    if name == "med":
        return get_med(nsamples, seed, seqlen, tokenizer, disentangle=disentangle)
