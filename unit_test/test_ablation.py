#!/usr/bin/env python
# -*- coding:utf-8 -*-
###
# Filename: /root/autodl-tmp/mol_prophecy/unit_test/test_ablation.py
# Path: /root/autodl-tmp/mol_prophecy/unit_test
# Created Date: Saturday, May 4th 2024, 9:33:57 pm
# Author: Qiong Zhou
# 
# Copyright (c) 2024 Your Company
###

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from omegaconf import OmegaConf
from ablation_experiment.AblationProphecy import LLMOnlyProphecy,ExpertOnlyProphecy
from torchdrug import data


def test_model_loading():
    cfg = OmegaConf.load("../config/config.yaml")
    if cfg.train.Ablation.experiment_model == 'bert' or cfg.train.Ablation.experiment_model == 'llama':
        mol_prophecy = LLMOnlyProphecy(cfg)
    elif cfg.train.Ablation.experiment_model == 'graph' or cfg.train.Ablation.experiment_model == 'sequence':
        mol_prophecy = ExpertOnlyProphecy(cfg)

    assert mol_prophecy is not None


def test_encoding():
    cfg = OmegaConf.load("../config/config.yaml")
    if cfg.train.Ablation.experiment_model == 'bert' or cfg.train.Ablation.experiment_model == 'llama':
        mol_prophecy = LLMOnlyProphecy(cfg)
    elif cfg.train.Ablation.experiment_model == 'graph' or cfg.train.Ablation.experiment_model == 'sequence':
        mol_prophecy = ExpertOnlyProphecy(cfg)

    with open("../data/test.json") as f:
        text_dict = json.load(f)

    text = text_dict["Insights"]
    smi = text_dict["Query SMILES"]

    embedding = mol_prophecy(smi, text)

    extra_feature_dim = cfg.Projection.physicochemical_properties_output_dim if cfg.Expert.physicochemical_properties else 0

    if cfg.train.Ablation.experiment_model == 'bert' or cfg.train.Ablation.experiment_model == 'llama':
        assert embedding.shape[
            -1] == cfg.Projection.llm_output_dim + extra_feature_dim
    elif cfg.train.Ablation.experiment_model == 'graph' or cfg.train.Ablation.experiment_model == 'sequence':
        assert embedding.shape[
            -1] == cfg.Projection.expert_output_dim + extra_feature_dim
    
