#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: /home/richard/projects/mol_prophecy/model/expert_graph.py
# Project: /home/richard/projects/mol_prophecy/model/mols
# Created Date: Thursday, January 25th 2024, 5:47:56 pm
# Author: Ruochi Zhang
# Email: zrc720@gmail.com
# -----
# Last Modified: Fri Jan 26 2024
# Modified By: Ruochi Zhang
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
from transformers import AutoTokenizer, AutoModelForMaskedLM



class SequenceExpert(nn.Module):

    def __init__(self,
                 model_path, tokenizer_path, padding, truncation,
                 max_length,output_hidden_states):
        super().__init__()
        # Load model directly

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForMaskedLM.from_pretrained(tokenizer_path)

        self.padding = padding
        self.truncation = truncation
        self.max_length = max_length
        self.output_hidden_states = output_hidden_states

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

        outputs = self.model(**encoding,output_hidden_states=self.output_hidden_states)

        return outputs
