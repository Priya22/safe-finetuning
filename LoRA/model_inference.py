import os, re, sys, json, csv, string, gzip, copy
import pandas as pd
import numpy as np
import argparse
# import unsloth
# from unsloth import FastLanguageModel

import torch
from transformers import TextStreamer
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# from vllm import SamplingParams, LLM



import logging
import time
from tqdm import tqdm

torch.cuda.empty_cache()

parser = argparse.ArgumentParser()
parser.add_argument('--config_path', required=True, help='config file for model inference') #config file with all variations -- have in separate module
parser.add_argument('--log_file', required=True, help='path to log file')


'''
### PROCESS ###
1. Load questions from SorryBench
    1.1 OR any other prompts, try to keep it general
2. get generations from model
    - Local/cloud model weights OR API-based models
3. Judge responses using Judge-LLM
    - local/cloud model weights OR API-based models
4. (optional) Evaluate Judge-LLM using human ground-truth
'''

class CandidateModel:
    def __init__(self, config):
        self.config = config
        logging.info("Config: {}".format(config))
        self.model = None
        self.tokenizer = None

        self.generate_func = None

        if config.model_source == 'api':
            pass
        
        # if config.lib == 'unsloth':

        #     self.generate_func = self.unsloth_inference
        
        else:
            self.generate_func = self.hf_inference

        self.model, self.tokenizer = self.load_model(self.config.model_path, self.config.lib)

    # def load_questions(self):
    #     '''
    #     load and return questions, answers
    #     '''
    #     fpath = self.config.get('file_path', './safetyBenchmarks/sorryBench/data/sorry-bench/question.jsonl')
    #     questions = sfb_common.load_questions(fpath, None, None)
    #     return questions

    def load_model(self, path=None, lib='hf'):
        '''
        load model weights and associated tokenizer (local or HF path)
        Works for default and LoRA-tuned models.
        '''
        if path is None:
            path = self.config.model_path
        config = self.config
        # if lib == 'unsloth':
        #     model, tokenizer = FastLanguageModel.from_pretrained(model_name=path, max_seq_length=self.config.max_seq_len, dtype=torch.float16, load_in_4bit=self.config.load_in_4bit, \
        #                                                         token = self.config.hf_token)
        #     model = FastLanguageModel.for_inference(model)

        # else: # assume hf | vllm errors toberesolved
        if config.load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_storage=torch.float16
                )
            model = AutoModelForCausalLM.from_pretrained(path, device_map='auto', \
                                                     quantization_config=quantization_config, torch_dtype=torch.float16)
        else:
            model = AutoModelForCausalLM.from_pretrained(path, device_map='auto')
        
        tokenizer = AutoTokenizer.from_pretrained(path, padding_side='left')
        tokenizer.pad_token = tokenizer.eos_token
        

        return model, tokenizer
        # pass

    def open_api_channel(self, model_family, api_key):
        '''
        OpenAI, Anthropic, Gemini, ...
        '''
        pass

    # def model_generate(self, ip_toks, temp, n_iters, max_new_toks, min_p, do_sample):
    #     '''
    #     TO DO: BATCH THIS PROCESS
    #     ip_toks: tokenized tensor
    #     '''
    #     ops = []
    #     for _ in range(n_iters):
    #         ops.append(self.model.generate(ip_toks, max_new_tokens = max_new_toks, temperature=temp, min_p=min_p, do_sample=do_sample, top_k=0))
    #     return ops

    def hf_inference(self, inputs):
        config = self.config
        gen_start = config.split_chars
        gen_end = config.end_of_text_chars

        def clean_text(op):
            return op.split(gen_start)[-1].split(gen_end)[0]

        device = self.model.device
        def tokenize(x):
            return self.tokenizer.apply_chat_template(x, tokenize=True, max_length=config.max_seq_len, add_generation_prompt=True, return_tensors='pt', padding=True, truncation=True).to(device)
        # input_ids = list(map(tokenize, inputs))

        temp = self.config.get('temperature', 1.0)
        num_iters = self.config.get('n_samples', 1)
        min_p = self.config.get('min_p', 0.1)
        # do_sample = self.config.get('do_sample', False)
        max_toks = self.config.get('max_new_toks', 512)

        do_sample = True
        batch_size = self.config.get('batch_size', 50)
        if batch_size is None:
            batch_size = 50
        # if temp > 0.00001:
        #     do_sample = False
        logging.info("Starting inference | {} instances, using HF with batch size {}".format(len(inputs), batch_size))
        st = time.time()
        # output_ids = list(map(lambda x: self.model_generate(x, temp, num_iters, max_toks, min_p, do_sample), input_ids)) # list of lists: n responses per input based on temperature and num_iters
        all_outputs = []
        with torch.no_grad():
            for ni in range(num_iters):
                outputs = []
                i = 0
                while i < len(inputs):
                    batch = inputs[i: i+batch_size]
                    inp_ids = tokenize(batch)
                    generated_ids = self.model.generate(inp_ids, temperature=temp, min_p=min_p, max_new_tokens=max_toks, do_sample=do_sample)
                    outputs.extend(self.tokenizer.batch_decode(generated_ids.detach().cpu().numpy()))
                    inp_ids = None
                    i = i + batch_size
                    if i%batch_size == 0:
                        print("Iteration {} | Done {}/{}".format(ni, i, len(inputs)))
                all_outputs.append([clean_text(s) for s in outputs])
                
        outputs = [[op[i] for op in all_outputs] for i in range(len(inputs))]
        return outputs

    def unsloth_inference(self, inputs):
        '''
        TO DO: BATCH THIS PROCESS        
        '''
        device = self.config.device
        def tokenize(x):
            return self.tokenizer.apply_chat_template(x, tokenize=False, add_generation_prompt=True) # first convert to template
            
        template_texts = list(map(tokenize, inputs))

        temp = self.config.get('temperature', 1.0)
        num_iters = self.config.get('n_samples', 1)
        min_p = self.config.get('min_p', 0.01)
        top_k = self.config.get('top_k', 50)
        # do_sample = self.config.get('do_sample', False)
        max_toks = self.config.get('max_new_toks', 1024)

        do_sample = True
        # if temp > 0.00001:
        #     do_sample = False
        logging.info("Starting inference..")
        sti = time.time()
        stb = time.time()
        # output_ids = list(map(lambda x: self.model_generate(x, temp, num_iters, max_toks, min_p, do_sample), input_ids)) # list of lists: n responses per input based on temperature and num_iters
        output_texts = []
        for i in range(0, len(template_texts)):
            ops = []
            input_ids = self.tokenizer(template_texts[i], return_tensors='pt').to("cuda")
            for _ in range(num_iters):
                opids=self.model.generate(**input_ids, max_new_tokens = max_toks, temperature=temp, min_p=min_p, do_sample=do_sample, top_k=top_k, use_cache=True)
                ops.append(self.tokenizer.decode(opids[0].detach().cpu().numpy()))
            # output_ids.append(ops)
            output_texts.append(ops)
            if i%self.config.batch_size == 0:
                logging.info("Input {}/{} done in {} seconds".format(i, len(inputs), time.time()-stb))
                stb = time.time()
                
        logging.info("Processed {} inputs in {} seconds".format(len(inputs), time.time()-sti))
        # out_texts = [[self.tokenizer.decode(x[0]) for x in l] for l in output_ids]
        return output_texts
        # pass
    
    def get_responses(self, inputs, input_ids=None):
        '''
        inputs: the main prompt to the model.
        '''

        # if not save_file:
        #     save_file = self.config.get('save_file')
        # model_id = self.config.model_id
        if not input_ids:
            input_ids = list(range(1, len(inputs)+1))
        # questions = self.load_questions()
        messages = [[{"role": "user", "content": x}] for x in inputs]
        outputs = self.generate_func(messages)

        split_chars = self.config.split_chars
        responses = [[x.rsplit(split_chars)[-1].strip() if isinstance(x, str) else None for x in qa] for qa in outputs]
        
        # if save_file:
        #     # save_file = os.path.join(save_file, "")

        #     with open(save_file, 'a+') as f:
        #         writer = csv.writer(f)
        #         writer.writerow(["model_id", "question_id"] +["answer_{}".format(i) for i in range(self.config.n_samples)])
        #         for qid, q, ans in zip(input_ids, inputs, responses):
        #             writer.writerow([model_id, qid. q] + ans)
        
        # logging.info("Outputs written to {}".format(save_file))
        # if return_vals:
        return input_ids, inputs, responses

# if __name__=='__main__':
#     args = parser.parse_args()
#     log_file = args.log_path
#     logging.basicConfig(filename='example.log', encoding='utf-8', level=logging.DEBUG, format='%(asctime)s %(message)s', filemode="a+")
