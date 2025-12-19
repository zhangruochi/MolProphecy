#!/usr/bin/env python
# -*- coding:utf-8 -*-
###
# Filename: /root/autodl-tmp/mol_prophecy/llama3_gen.py
# Path: /root/autodl-tmp/mol_prophecy
# Created Date: Sunday, June 9th 2024, 12:30:27 pm
# Author: Qiong Zhou
# 
# Copyright (c) 2024 Your Company
###

# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import LlamaForSequenceClassification, LlamaTokenizer

import os
import torch
import numpy as np
import json
from omegaconf import OmegaConf

cfg = OmegaConf.load("./config/config.yaml")

dataset_name = cfg.train.dataset  
dataset_path = "./data/{}".format(dataset_name)

if not os.path.exists(os.path.join(dataset_path, "llama3_np")):
    os.makedirs(os.path.join(dataset_path, "llama3_np"), exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained("/mnt/data/zhouqiong/hugging-hub/pretrained/Meta-Llama-3-8B")
model = AutoModelForCausalLM.from_pretrained("/mnt/data/zhouqiong/hugging-hub/pretrained/Meta-Llama-3-8B", output_hidden_states=True)
tokenizer.add_special_tokens(
        {

            "pad_token": "<PAD>",
        }
    )
model.resize_token_embeddings(model.config.vocab_size + 1)

map_dict = {}
attention_dict = {}
with open("{}/map.txt".format(dataset_path), 'r') as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip()
        if line:
            line_list = line.split('\t')
            with open(line_list[1], 'r') as file:
                data = json.load(file)
                insight_text = str(data["Insights"])
            # encode text and get best hidden state embedding
            with torch.no_grad():
                encoding = tokenizer(insight_text,padding='max_length',truncation=True,max_length=cfg.LLMs.llama.max_length,return_tensors='pt')
                model.eval()
                outputs = model(**encoding)
                best_hidden_states = outputs.hidden_states[12].numpy()  
            filename = os.path.basename(line_list[1])
            np.save(os.path.join(dataset_path, "llama3_np", f'{os.path.splitext(filename)[0]}.npy'), best_hidden_states)
            #save attention mask
            np.save(os.path.join(dataset_path, "llama3_np", f'{os.path.splitext(filename)[0]}_attention_mask.npy'), encoding['attention_mask'].numpy())
            
            smiles = line_list[0]
            map_dict[smiles] = os.path.join(dataset_path, "llama3_np", f'{os.path.splitext(filename)[0]}.npy')
            attention_dict[smiles] = os.path.join(dataset_path, "llama3_np", f'{os.path.splitext(filename)[0]}_attention_mask.npy')
            print(filename,smiles,insight_text)
            print('-'*30)

with open(os.path.join(dataset_path, 'llama2npy_map.json'), 'w') as f:
    json.dump(map_dict, f)

#save attention mask
with open(os.path.join(dataset_path, 'llama2npy_attention_mask.json'), 'w') as f:
    json.dump(attention_dict, f)