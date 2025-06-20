#!/usr/bin/env python
# -*- coding:utf-8 -*-
###
# Filename: /root/autodl-tmp/mol_prophecy/mytorchdrug/PropertyPrediction4Explain.py
# Path: /root/autodl-tmp/mol_prophecy/mytorchdrug
# Created Date: Monday, May 20th 2024, 3:48:44 pm
# Author: Qiong Zhou
#
# Copyright (c) 2024 Your Company
###
import os
import pandas as pd
import math
import torch
from torchdrug import tasks, layers
from collections import defaultdict
from torchdrug.core import Registry as R


@R.register("tasks.PropertyPrediction4Explain")
class PropertyPrediction4Explain(tasks.PropertyPrediction):
    eps = 1e-10
    _option_members = {"task", "criterion", "metric"}

    def __init__(self,
                 model,
                 task=(),
                 criterion="mse",
                 metric=("mae", "rmse"),
                 default_metric="rmse",
                 num_mlp_layer=1,
                 normalization=True,
                 num_class=None,
                 mlp_batch_norm=False,
                 mlp_dropout=0,
                 graph_construction_model=None,
                 verbose=0):
        super().__init__(model=model,
                         task=task,
                         criterion=criterion,
                         metric=metric,
                         num_mlp_layer=num_mlp_layer,
                         normalization=normalization,
                         num_class=num_class,
                         mlp_batch_norm=mlp_batch_norm,
                         mlp_dropout=mlp_dropout,
                         graph_construction_model=graph_construction_model,
                         verbose=verbose)
        self.model = model
        self.task = task
        self.criterion = criterion
        self.metric = metric
        self.default_metric = default_metric
        self.num_mlp_layer = num_mlp_layer
        # For classification tasks, we disable normalization tricks.
        self.normalization = normalization and ("ce" not in criterion) and (
            "bce" not in criterion)
        self.num_class = (num_class, ) if isinstance(num_class,
                                                     int) else num_class
        self.mlp_batch_norm = mlp_batch_norm
        self.mlp_dropout = mlp_dropout
        self.graph_construction_model = graph_construction_model
        self.verbose = verbose

        # load_dotenv()
        # openai.api_key = os.getenv("OPENAI_API_KEY")
        # self.cfg = OmegaConf.load("config/config.yaml")

    def preprocess(self, train_set, valid_set, test_set):
        """
        Compute the mean and derivation for each task on the training set.
        """
        values = defaultdict(list)
        for sample in train_set:
            if not sample.get("labeled", True):
                continue
            for task in self.task:
                if not math.isnan(sample[task]):
                    values[task].append(sample[task])
        mean = []
        std = []
        weight = []
        num_class = []
        for task, w in self.task.items():
            value = torch.tensor(values[task])
            mean.append(value.float().mean())
            std.append(value.float().std())
            weight.append(w)
            if value.ndim > 1:
                num_class.append(value.shape[1])
            elif value.dtype == torch.long:
                task_class = value.max().item()
                if task_class == 1 and "bce" in self.criterion:
                    num_class.append(1)
                else:

                    num_class.append(task_class + 1)
            else:
                num_class.append(1)

        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float))
        self.register_buffer("std", torch.as_tensor(std, dtype=torch.float))
        self.register_buffer("weight",
                             torch.as_tensor(weight, dtype=torch.float))
        self.num_class = self.num_class or num_class

        hidden_dims = [self.model.output_dim] * (self.num_mlp_layer - 1)
        self.mlp = layers.MLP(self.model.output_dim,
                              hidden_dims + [sum(self.num_class)],
                              batch_norm=self.mlp_batch_norm,
                              dropout=self.mlp_dropout)

    def predict(self, x, edge_index, edge_attr, batch):

        embedding, _ = self.model(x, edge_index, edge_attr, batch)
        pred = self.mlp(embedding)

        if self.normalization:
            pred = pred * self.std + self.mean

        return pred
    
    def predict4llm_x(self, input_ids, attention_mask=None, token_type_ids=None):

        embedding, _ = self.model(input_ids, attention_mask=attention_mask, token_type_ids=None)
        pred = self.mlp(embedding)

        if self.normalization:
            pred = pred * self.std + self.mean

        return pred
