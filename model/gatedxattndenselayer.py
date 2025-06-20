#!/usr/bin/env python
# -*- coding:utf-8 -*-
###
# Filename: /root/autodl-tmp/mol_prophecy/model/gatedxattndenselayer.py
# Path: /root/autodl-tmp/mol_prophecy/model
# Created Date: Wednesday, July 3rd 2024, 10:26:31 am
# Author: Qiong Zhou
# 
# Copyright (c) 2024 Your Company
###
import torch
import torch.nn as nn

class SquaredReLU(nn.Module):
    def forward(self, x):
        return torch.relu(x) ** 2

class GatedXattnDenseLayer(nn.Module):
    def __init__(self, d_model, num_heads,dropout=0):
        super(GatedXattnDenseLayer, self).__init__()
        self.d_model = d_model
        
        self.cross_attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, batch_first=True,dropout=dropout)
        self.ffw = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU()
        )
        self.frozen_ffw = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU()
        )
        
        self.frozen_self_attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, batch_first=True,dropout=dropout)
        
        self.alpha_xattn = nn.Parameter(torch.tensor(0.0))
        self.alpha_dense = nn.Parameter(torch.tensor(0.0))

        # self._init_frozen_layers()

    def _init_frozen_layers(self):
        
        # frozen parameters
        for param in self.frozen_ffw.parameters():
            param.requires_grad = False
        
        for param in self.frozen_self_attention.parameters():
            param.requires_grad = False

    def forward(self, y, x,y_mask,x_mask):
        # Convert x_mask to boolean and invert it , 0 for padding
        x_mask = (x_mask == 0)
        cross_attn_output, _ = self.cross_attention(y, x, x,key_padding_mask=x_mask)
        y = y + torch.tanh(self.alpha_xattn) * cross_attn_output
        # gated ffw
        y = y + torch.tanh(self.alpha_dense) * self.ffw(y)
        # y_mask is boolean, False for padding
        y_mask = (y_mask == False)
        self_attn_output, _ = self.frozen_self_attention(y, y, y,key_padding_mask=y_mask)
        y = y + self_attn_output
        y = y + self.frozen_ffw(y)
        
        return y




