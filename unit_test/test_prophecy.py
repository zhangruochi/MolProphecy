#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: /home/richard/projects/mol_prophecy/unit_test/test_prophecy.py
# Project: /home/richard/projects/mol_prophecy/unit_test
# Created Date: Thursday, January 25th 2024, 10:04:03 pm
# Author: Ruochi Zhang
# Email: zrc720@gmail.com
# -----
# Last Modified: Tue May 07 2024
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

import json
from omegaconf import OmegaConf
from mol_prophecy.model.prophecy import Prophecy
from torchdrug import data


def test_model_loading():
    cfg = OmegaConf.load("../config/config.yaml")
    mol_prophecy = Prophecy(cfg)

    assert mol_prophecy is not None


def test_encoding():
    cfg = OmegaConf.load("../config/config.yaml")
    mol_prophecy = Prophecy(cfg)

    with open("../data/test.json") as f:
        text_dict = json.load(f)

    text = text_dict["Insights"]
    smi = text_dict["Query SMILES"]

    embedding,_ = mol_prophecy(smi, text)

    extra_feature_dim = cfg.Projection.physicochemical_properties_output_dim if cfg.Expert.physicochemical_properties else 0

    if cfg.train.fusion_approach == 'concat':
        assert embedding.shape[
            -1] == cfg.Projection.expert_output_dim + cfg.Projection.llm_output_dim + extra_feature_dim
    elif cfg.train.fusion_approach == 'tensor_fusion':
        assert embedding.shape[-1] == (cfg.Projection.expert_output_dim +
                                       1) * (cfg.Projection.llm_output_dim +
                                             1) + extra_feature_dim
    elif cfg.train.fusion_approach == 'bilinear_fusion':
        assert embedding.shape[-1] == (cfg.Projection.expert_output_dim) * (
            cfg.Projection.llm_output_dim) + extra_feature_dim
