#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: /home/richard/projects/mol_prophecy/model/llms_embedding.py
# Project: /home/richard/projects/mol_prophecy/model/llms
# Created Date: Thursday, January 25th 2024, 11:11:40 am
# Author: Ruochi Zhang
# Email: zrc720@gmail.com
# -----
# Last Modified: Fri Jul 05 2024
# Modified By: Qiong Zhou
# -----
# Copyright (c) 2024 Bodkin World Domination Enterprises
#
# MIT License
#
# Copyright (c) 2024 Ruochi Zhang
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
# of the Software, and to permit persons to whom the Software is furnished to do
# so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# -----
###

import torch
from torch import nn
from transformers import BertTokenizer, BertModel
from transformers import AutoTokenizer, AutoModelForCausalLM

class BertEmbedding(nn.Module):

    def __init__(self, model_path, tokenizer_path, padding, truncation,
                 max_length):
        super().__init__()
        self.model = BertModel.from_pretrained(model_path)
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
        self.padding = padding
        self.truncation = truncation
        self.max_length = max_length

    def tokenize(self, text):
        encoding = self.tokenizer(text=text,
                                  padding=self.padding,
                                  truncation=self.truncation,
                                  max_length=self.max_length,
                                  return_tensors='pt')
        return encoding

    def convert_ids_to_tokens(self, ids):
        return self.tokenizer.convert_ids_to_tokens(ids)

    def decode(self, ids):
        return self.tokenizer.decode(ids,
                                     skip_special_tokens=True,
                                     clean_up_tokenization_spaces=True)

    def forward(self, text):
        if isinstance(text, str):
            text = [text]

        encoding = self.tokenize(text=text).to(self.model.device)
        outputs = self.model(**encoding)

        return outputs,encoding['attention_mask']



class LlamaEmbedding(nn.Module):

    def __init__(self, model_path, tokenizer_path, padding, truncation,
                 max_length):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(model_path,output_hidden_states=True)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.tokenizer.add_special_tokens(
                {

                    "pad_token": "<PAD>",
                }
            )
        self.model.resize_token_embeddings(self.model.config.vocab_size + 1)
        self.padding = padding
        self.truncation = truncation
        self.max_length = max_length

    def tokenize(self, text):
        encoding = self.tokenizer(text=text,
                                  padding=self.padding,
                                  truncation=self.truncation,
                                  max_length=self.max_length,
                                  return_tensors='pt')
        return encoding

    def convert_ids_to_tokens(self, ids):
        return self.tokenizer.convert_ids_to_tokens(ids)

    def decode(self, ids):
        return self.tokenizer.decode(ids,
                                     skip_special_tokens=True,
                                     clean_up_tokenization_spaces=True)

    def forward(self, text):
        if isinstance(text, str):
            text = [text]

        encoding = self.tokenize(text=text).to(self.model.device)
        outputs = self.model(**encoding)

        return outputs,encoding['attention_mask']