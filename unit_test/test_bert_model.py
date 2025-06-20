#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: /home/richard/projects/mol_prophecy/unit_test/test_llm.py
# Project: /home/richard/projects/mol_prophecy/unit_test
# Created Date: Thursday, January 25th 2024, 3:16:05 pm
# Author: Ruochi Zhang
# Email: zrc720@gmail.com
# -----
# Last Modified: Fri Apr 26 2024
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

import os
import json
from omegaconf import OmegaConf
from mol_prophecy.model.llms.llm_bert import BertEmbedding

# load .env file
from dotenv import load_dotenv

load_dotenv()


def test_encoding():
    cfg = OmegaConf.load("../config/config.yaml")
    llm = BertEmbedding(
        os.path.join(os.getenv("PRETRAINED_ROOT"), cfg.LLMs.bert.model),
        os.path.join(os.getenv("PRETRAINED_ROOT"),
                     cfg.LLMs.bert.tokenizer), cfg.LLMs.bert.padding,
        cfg.LLMs.bert.truncation, cfg.LLMs.bert.max_length)
    with open("../data/test.json") as f:
        text_dict = json.load(f)

    text = text_dict["Insights"]
    encoding = llm.tokenize(text)

    assert encoding["input_ids"].shape[-1] == cfg.LLMs.bert.max_length
    assert encoding["attention_mask"].shape[-1] == cfg.LLMs.bert.max_length
    assert encoding["token_type_ids"].shape[-1] == cfg.LLMs.bert.max_length


def test_convert_ids_to_tokens():
    cfg = OmegaConf.load("../config/config.yaml")
    llm = BertEmbedding(
        os.path.join(os.getenv("PRETRAINED_ROOT"), cfg.LLMs.bert.model),
        os.path.join(os.getenv("PRETRAINED_ROOT"),
                     cfg.LLMs.bert.tokenizer), cfg.LLMs.bert.padding,
        cfg.LLMs.bert.truncation, cfg.LLMs.bert.max_length)
    with open("../data/test.json") as f:
        text_dict = json.load(f)

    text = text_dict["Insights"]
    encoding = llm.tokenize(text)["input_ids"].numpy().tolist()[0]
    decoding = llm.convert_ids_to_tokens(encoding)

    assert len(encoding) == len(decoding)
    assert decoding[0] == "[CLS]"


def test_decoding():
    cfg = OmegaConf.load("../config/config.yaml")
    llm = BertEmbedding(
        os.path.join(os.getenv("PRETRAINED_ROOT"), cfg.LLMs.bert.model),
        os.path.join(os.getenv("PRETRAINED_ROOT"),
                     cfg.LLMs.bert.tokenizer), cfg.LLMs.bert.padding,
        cfg.LLMs.bert.truncation, cfg.LLMs.bert.max_length)
    with open("../data/test.json") as f:
        text_dict = json.load(f)

    text = text_dict["Insights"]
    encoding = llm.tokenize(text)["input_ids"].numpy().tolist()[0]
    decoded_text = llm.decode(encoding)

    print("")
    print(text)
    print(decoded_text)


def test_embedding():
    cfg = OmegaConf.load("../config/config.yaml")
    llm = BertEmbedding(
        os.path.join(os.getenv("PRETRAINED_ROOT"), cfg.LLMs.bert.model),
        os.path.join(os.getenv("PRETRAINED_ROOT"),
                     cfg.LLMs.bert.tokenizer), cfg.LLMs.bert.padding,
        cfg.LLMs.bert.truncation, cfg.LLMs.bert.max_length)
    with open("../data/test.json") as f:
        text_dict = json.load(f)

    text = text_dict["Insights"]
    embedding = llm(text).last_hidden_state

    assert (embedding.shape[1],
            embedding.shape[2]) == (cfg.LLMs.bert.max_length, 768)
