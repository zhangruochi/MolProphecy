#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: /home/richard/projects/mol_prophecy/model/expert_graph.py
# Project: /home/richard/projects/mol_prophecy/model/mols
# Created Date: Thursday, January 25th 2024, 5:47:56 pm
# Author: Ruochi Zhang
# Email: zrc720@gmail.com
# -----
# Last Modified: Sun Jul 07 2024
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

from torch import nn
from .molclr import GINet, GraphTokenizer, GINet4Explain
import torch
import os
from typing import List


class GraphExpert(nn.Module):

    def __init__(self,
                 pretrained_model_path,
                 emb_dim,
                 feat_dim,
                 edge_feature_dim,
                 num_graph_layers,
                 drop_ratio,
                 cfg,
                 batch_norm=True,
                 readout="mean"):
        super().__init__()

        # self.gin_model = models.GIN(input_dim=node_feature_dim,
        #                             hidden_dims=hidden_dims,
        #                             edge_input_dim=edge_feature_dim,
        #                             batch_norm=batch_norm,
        #                             readout=readout)

        self.gnn = GINet(emb_dim=emb_dim,
                         num_layer=num_graph_layers,
                         feat_dim=feat_dim,
                         drop_ratio=drop_ratio,
                         cfg=cfg,
                         pool=readout)

        state_dict = torch.load(pretrained_model_path, map_location="cpu")

        for key in list(state_dict.keys()):
            # pass the weights of the output layer
            if key.startswith("out_lin"):
                state_dict.pop(key)

        self.gnn.load_state_dict(state_dict, strict=True)
        print("Loaded pre-trained graph model with success.")

        self.tokenizer = GraphTokenizer()

    def forward(self, smiles: List, all_loss=None, metric=None):

        graph_data = self.tokenizer.batch_encode_plus(smiles).to(self.device)

        # output = self.gin_model(graph_structure,
        #                         node_features,
        #                         all_loss=all_loss,
        #                         metric=metric)
        graph_embeds,graph_mask,h_analysis = self.gnn(graph_data)

        return {"graph_feature": graph_embeds,"graph_mask": graph_mask,'h_analysis':h_analysis}

    @property
    def device(self):
        return next(self.parameters()).device


class GraphExpert4Explain(nn.Module):

    def __init__(self,
                 pretrained_model_path,
                 emb_dim,
                 feat_dim,
                 edge_feature_dim,
                 num_graph_layers,
                 drop_ratio,
                 batch_norm=True,
                 readout="mean"):
        super().__init__()

        self.gnn = GINet4Explain(emb_dim=emb_dim,
                                 num_layer=num_graph_layers,
                                 feat_dim=feat_dim,
                                 drop_ratio=drop_ratio,
                                 pool=readout)

        state_dict = torch.load(pretrained_model_path, map_location="cpu")

        for key in list(state_dict.keys()):
            # pass the weights of the output layer
            if key.startswith("out_lin"):
                state_dict.pop(key)

        self.gnn.load_state_dict(state_dict, strict=True)
        print("Loaded pre-trained graph model with success.")

    def forward(self,
                x,
                edge_index,
                edge_attr,
                batch,
                all_loss=None,
                metric=None):

        graph_embeds = self.gnn(x, edge_index, edge_attr, batch)

        return {"graph_feature": graph_embeds}

    @property
    def device(self):
        return next(self.parameters()).device
