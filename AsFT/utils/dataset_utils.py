# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import torch

from functools import partial

from ft_datasets import (
    get_alpaca_dataset,
    get_pure_bad_dataset,
    get_gsm8k_dataset,
    get_agnews_dataset,
    get_agnews_dataset,
    get_SST2_dataset,
)
from typing import Optional


DATASET_PREPROC = {
    "alpaca_dataset": partial(get_alpaca_dataset, max_words=512),
    "gsm8k_dataset": partial(get_gsm8k_dataset, max_words=512),
    "agnews_dataset": partial(get_agnews_dataset, max_words=512),
    "pure_bad_dataset": partial(get_pure_bad_dataset, max_words=480),
    "SST2_dataset": partial(get_SST2_dataset, max_words=512),
}


def get_preprocessed_dataset(
    tokenizer, dataset_config, split: str = "train"
) -> torch.utils.data.Dataset:
    if not dataset_config.dataset in DATASET_PREPROC:
        raise NotImplementedError(f"{dataset_config.dataset} is not (yet) implemented")

    def get_split():
        return (
            dataset_config.train_split
            if split == "train"
            else dataset_config.test_split
        )
    
    return DATASET_PREPROC[dataset_config.dataset](
        dataset_config,
        tokenizer,
        get_split(),
    )

class StandardDataset:

  def __init__(self, text_ds, tokenizer, max_words=30):
        self.ann = text_ds
        self.max_words = max_words
        self.tokenizer = tokenizer

  def __len__(self):
      return len(self.ann)

  def __getitem__(self, index):
      IGNORE_INDEX = -100  # The default setting in CrossEntropyLoss

      ann = self.ann[index]
      prompt_enc = self.tokenizer.apply_chat_template([ann['prompt'][0]], tokenize=True, add_generation_prompt=False, \
                                                      dtype=torch.int64, return_tensors="pt")[0]

      example = self.tokenizer.apply_chat_template([ann['prompt'][0], ann['completion'][0]], tokenize=True, \
                                                   add_generation_prompt=False, truncation=True, padding='max_length', \
                                                   max_length=self.max_words, return_tensors="pt", dtype=torch.int64)[0]
      example[example.eq(0)] = -1
      
      labels = copy.deepcopy(example)
      labels[: len(prompt_enc)] = -1
      example_mask = example.ge(0)
      label_mask = labels.ge(0)
      example[~example_mask] = 0
      labels[~label_mask] = IGNORE_INDEX
      example_mask = example_mask.float()
      label_mask = label_mask.float()

      return {
          "input_ids": example,
          "labels": labels,
          "attention_mask":example_mask,
      }