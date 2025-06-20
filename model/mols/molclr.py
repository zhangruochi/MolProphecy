#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: /home/richard/projects/pep_interaction/model/molclr.py
# Project: /home/richard/projects/mol_prophecy/model/mols
# Created Date: Monday, March 4th 2024, 11:28:38 am
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

import os

import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool
from omegaconf import OmegaConf
from typing import List

from rdkit import Chem
from rdkit.Chem.rdchem import HybridizationType
from rdkit.Chem.rdchem import BondType as BT
from rdkit.Chem import AllChem
from torch_geometric.data import Data, Batch

# reference from https://github.com/yuyangw/MolCLR/

num_atom_type = 119  # including the extra mask tokens
num_chirality_tag = 3

num_bond_type = 5  # including aromatic and self-loop edge
num_bond_direction = 3

ATOM_LIST = list(range(1, 119))
CHIRALITY_LIST = [
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
    Chem.rdchem.ChiralType.CHI_OTHER
]
BOND_LIST = [BT.SINGLE, BT.DOUBLE, BT.TRIPLE, BT.AROMATIC, BT.DATIVE]
BONDDIR_LIST = [
    Chem.rdchem.BondDir.NONE, Chem.rdchem.BondDir.ENDUPRIGHT,
    Chem.rdchem.BondDir.ENDDOWNRIGHT
]


class GINEConv(MessagePassing):

    def __init__(self, emb_dim):
        super(GINEConv, self).__init__()
        self.mlp = nn.Sequential(nn.Linear(emb_dim, 2 * emb_dim), nn.ReLU(),
                                 nn.Linear(2 * emb_dim, emb_dim))
        self.edge_embedding1 = nn.Embedding(num_bond_type, emb_dim)
        self.edge_embedding2 = nn.Embedding(num_bond_direction, emb_dim)
        nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

    def forward(self, x, edge_index, edge_attr):
        # add self loops in the edge space
        edge_index = add_self_loops(edge_index, num_nodes=x.size(0))[0]

        # add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), 2)
        self_loop_attr[:, 0] = 4  #bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(
            edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)

        edge_embeddings = self.edge_embedding1(
            edge_attr[:, 0]) + self.edge_embedding2(edge_attr[:, 1])

        return self.propagate(edge_index, x=x, edge_attr=edge_embeddings)

    def message(self, x_j, edge_attr):
        return x_j + edge_attr

    def update(self, aggr_out):
        return self.mlp(aggr_out)


class GINet(nn.Module):
    """
    Args:
        num_layer (int): the number of GNN layers
        emb_dim (int): dimensionality of embeddings
        max_pool_layer (int): the layer from which we use max pool rather than add pool for neighbor aggregation
        drop_ratio (float): dropout rate
        gnn_type: gin, gcn, graphsage, gat
    Output:
        node representations
    """

    def __init__(self,
                 num_layer=5,
                 emb_dim=300,
                 feat_dim=256,
                 drop_ratio=0,
                 cfg=None,
                 pool='mean'):
        super(GINet, self).__init__()
        self.num_layer = num_layer
        self.emb_dim = emb_dim
        self.feat_dim = feat_dim
        self.drop_ratio = drop_ratio
        self.cfg = cfg

        self.x_embedding1 = nn.Embedding(num_atom_type, emb_dim)
        self.x_embedding2 = nn.Embedding(num_chirality_tag, emb_dim)
        nn.init.xavier_uniform_(self.x_embedding1.weight.data)
        nn.init.xavier_uniform_(self.x_embedding2.weight.data)

        # List of MLPs
        self.gnns = nn.ModuleList()
        for layer in range(num_layer):
            self.gnns.append(GINEConv(emb_dim))

        # List of batchnorms
        self.batch_norms = nn.ModuleList()
        for layer in range(num_layer):
            self.batch_norms.append(nn.BatchNorm1d(emb_dim))

        if pool == 'mean':
            self.pool = global_mean_pool
        elif pool == 'max':
            self.pool = global_max_pool
        elif pool == 'add':
            self.pool = global_add_pool

        self.feat_lin = nn.Linear(self.emb_dim, self.feat_dim)

        # self.out_lin = nn.Sequential(
        #     nn.Linear(self.feat_dim, self.feat_dim), nn.ReLU(inplace=True),
        #     nn.Linear(self.feat_dim, self.feat_dim // 2))

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr

        h = self.x_embedding1(x[:, 0]) + self.x_embedding2(x[:, 1])

        for layer in range(self.num_layer):
            h = self.gnns[layer](h, edge_index, edge_attr)
            h = self.batch_norms[layer](h)
            if layer == self.num_layer - 1:
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h),
                              self.drop_ratio,
                              training=self.training)
        padding_masks = []
        if self.cfg.train.fusion_approach == 'attention':
            if self.cfg.train.Ablation.is_ablation == True and self.cfg.train.Ablation.experiment_model == 'graph':
                h = self.pool(h, data.batch)
                h_analysis = h
            else:
                data_batch = data.batch
                # this is a variant for model analysis
                h_analysis = self.pool(h, data.batch)
                sequence_length = self.cfg.LLMs.bert.max_length
                sample_embeddings = {}
                for idx in range(data_batch.max().item() + 1):
                    sample_embeddings[idx] = h[data_batch == idx]
                padded_embeddings = []
                for idx in sample_embeddings:
                    emb = sample_embeddings[idx]
                    mask = torch.ones(sequence_length, dtype=torch.bool,device=emb.device) 
                    if emb.size(0) < sequence_length:
                        padded_emb = F.pad(emb, (0, 0, 0, sequence_length - emb.size(0)), "constant", 0)
                        mask[emb.size(0):] = False
                    else:
                        padded_emb = emb[:sequence_length]
                        mask = mask[:sequence_length] 
                    padded_embeddings.append(padded_emb.unsqueeze(0))
                    padding_masks.append(mask.unsqueeze(0))
                h = torch.cat(padded_embeddings, dim=0)
                padding_masks = torch.cat(padding_masks, dim=0)
        else:
            h = self.pool(h, data.batch)
            h_analysis = h
        h = self.feat_lin(h)
        # out = self.out_lin(h)
        return h,padding_masks,h_analysis


class GraphTokenizer():

    def __init__(self):
        pass

    def tokenize(self, smiles: str):
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)

        N = mol.GetNumAtoms()
        M = mol.GetNumBonds()

        type_idx = []
        chirality_idx = []
        atomic_number = []
        for atom in mol.GetAtoms():
            type_idx.append(ATOM_LIST.index(atom.GetAtomicNum()))
            chirality_idx.append(CHIRALITY_LIST.index(atom.GetChiralTag()))
            atomic_number.append(atom.GetAtomicNum())

        x1 = torch.tensor(type_idx, dtype=torch.long).view(-1, 1)
        x2 = torch.tensor(chirality_idx, dtype=torch.long).view(-1, 1)
        x = torch.cat([x1, x2], dim=-1)

        row, col, edge_feat = [], [], []
        for bond in mol.GetBonds():
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            row += [start, end]
            col += [end, start]
            edge_feat.append([
                BOND_LIST.index(bond.GetBondType()),
                BONDDIR_LIST.index(bond.GetBondDir())
            ])
            edge_feat.append([
                BOND_LIST.index(bond.GetBondType()),
                BONDDIR_LIST.index(bond.GetBondDir())
            ])

        edge_index = torch.tensor([row, col], dtype=torch.long)
        edge_attr = torch.tensor(np.array(edge_feat), dtype=torch.long)

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        return data

    def batch_encode_plus(self, smiles: List):
        data_batch = []
        for smi in smiles:
            data_batch.append(self.tokenize(smi))

        return Batch.from_data_list(data_batch)


class GINet4Explain(nn.Module):

    def __init__(self,
                 num_layer=5,
                 emb_dim=300,
                 feat_dim=256,
                 drop_ratio=0,
                 pool='mean'):
        super(GINet4Explain, self).__init__()
        self.num_layer = num_layer
        self.emb_dim = emb_dim
        self.feat_dim = feat_dim
        self.drop_ratio = drop_ratio

        self.x_embedding1 = nn.Embedding(num_atom_type, emb_dim)
        self.x_embedding2 = nn.Embedding(num_chirality_tag, emb_dim)
        nn.init.xavier_uniform_(self.x_embedding1.weight.data)
        nn.init.xavier_uniform_(self.x_embedding2.weight.data)

        self.gnns = nn.ModuleList()
        for layer in range(num_layer):
            self.gnns.append(GINEConv(emb_dim))

        self.batch_norms = nn.ModuleList()
        for layer in range(num_layer):
            self.batch_norms.append(nn.BatchNorm1d(emb_dim))

        if pool == 'mean':
            self.pool = global_mean_pool
        elif pool == 'max':
            self.pool = global_max_pool
        elif pool == 'add':
            self.pool = global_add_pool

        self.feat_lin = nn.Linear(self.emb_dim, self.feat_dim)

    def forward(self, h, edge_index, edge_attr, batch):
        for layer in range(self.num_layer):
            h = self.gnns[layer](h, edge_index, edge_attr)
            h = self.batch_norms[layer](h)
            if layer == self.num_layer - 1:
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h),
                              self.drop_ratio,
                              training=self.training)

        h = self.pool(h, batch)
        h = self.feat_lin(h)
        return h

    def get_node_embedding(self, x):
        h = self.x_embedding1(x[:, 0].long()) + self.x_embedding2(x[:,
                                                                    1].long())
        return h
