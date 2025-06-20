#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###

import os
import sys
import torch
import json
import numpy as np
from torch import nn
from torchdrug import layers
from torch.nn.parameter import Parameter

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.mols.graph_expert import GraphExpert, GraphExpert4Explain
from model.llms.llm_bert import BertEmbedding,LlamaEmbedding
from model.seqs.sequence_expert import SequenceExpert

from model.prophecy import Prophecy


class LLMOnlyProphecy(Prophecy):

    def __init__(self, cfg):
        super().__init__(cfg)
        self.cfg = cfg

        if cfg.train.Ablation.experiment_model == 'bert':
            self.llm = BertEmbedding(
                os.path.join(os.getenv("PRETRAINED_ROOT"),
                             cfg.LLMs.bert.model),
                os.path.join(os.getenv("PRETRAINED_ROOT"),
                             cfg.LLMs.bert.tokenizer), cfg.LLMs.bert.padding,
                cfg.LLMs.bert.truncation, cfg.LLMs.bert.max_length)
            # LLMs projection layer
            self.llm_projection = layers.MLP(cfg.LLMs.bert.output_dim,
                                             cfg.Projection.llm_output_dim)
        elif cfg.train.Ablation.experiment_model == 'llama':
            if cfg.LLMs.llama.use_cache == False:
                self.llm = LlamaEmbedding(
                    os.path.join(os.getenv("PRETRAINED_ROOT"),
                                cfg.LLMs.llama.model),
                    os.path.join(os.getenv("PRETRAINED_ROOT"),
                                cfg.LLMs.llama.tokenizer), cfg.LLMs.llama.padding,
                    cfg.LLMs.llama.truncation, cfg.LLMs.llama.max_length)
            else:
                # load cache npy file
                with open(os.path.join("./data/{}/llama2npy_map.json".format(self.cfg.train.dataset)), 'r') as f:
                    self.llama_cache_dict = json.load(f)
                # load cache attention mask np file
                with open(os.path.join("./data/{}/llama2npy_attention_mask.json".format(self.cfg.train.dataset)), 'r') as f:
                    self.llama_cache_mask_dict = json.load(f)

            self.llm_projection = layers.MLP(cfg.LLMs.llama.output_dim,
                                             cfg.Projection.llm_output_dim)
        if cfg.train.Ablation.experiment_model == 'bert':
            for param in self.llm.parameters():
                param.requires_grad = False
        elif cfg.train.Ablation.experiment_model == 'llama':
            if cfg.LLMs.llama.use_cache == False:
                for param in self.llm.parameters():
                    param.requires_grad = False
        

        self.output_dim = cfg.Projection.llm_output_dim

    def forward(self, smiles, mol_descs, all_loss=None, metric=None):
        if isinstance(mol_descs, str):
            mol_descs = [mol_descs]
        if isinstance(smiles, str):
            smiles = [smiles]

        if self.cfg.train.Ablation.experiment_model == 'bert':
            text_output,_ = self.llm(mol_descs)
            text_feature = text_output.last_hidden_state[:, 0, :].squeeze(1)
        elif self.cfg.train.Ablation.experiment_model == 'llama':
            if self.cfg.LLMs.llama.use_cache == False:
                text_output,mask_attn = self.llm(mol_descs)
                embedding = list(text_output.hidden_states)
                hidden_states = embedding[12]
                text_feature = hidden_states[:,-1,:]
            else:
                #load cache npy file
                llama_cache_dict = self.llama_cache_dict
                embeddings = []
                for smiles_i in smiles:
                    file_path = llama_cache_dict.get(smiles_i, None)
                    embedding = np.load(file_path)
                    mask_file_path = self.llama_cache_mask_dict.get(smiles_i, None)
                    mask_attn = np.load(mask_file_path)
                    non_padded_embedding = embedding[:,-1,:]

                    embeddings.append(non_padded_embedding)
                text_feature = torch.tensor(np.concatenate(embeddings, axis=0),device=self.device)

        text_embeding = self.llm_projection(text_feature)

        return text_embeding, [text_feature]


class ExpertOnlyProphecy(Prophecy):

    def __init__(self, cfg):
        super().__init__(cfg)
        self.cfg = cfg

        # expert and llm
        if self.cfg.train.Ablation.experiment_model == 'graph':
            self.graph_model = GraphExpert(
                pretrained_model_path=os.path.join(
                    os.getenv("PRETRAINED_ROOT"), cfg.Expert.graph.model,
                    "model.pth"),
                feat_dim=cfg.Expert.graph.feat_dim,
                emb_dim=cfg.Expert.graph.emb_dim,
                edge_feature_dim=None,
                num_graph_layers=cfg.Expert.graph.num_graph_layers,
                drop_ratio=cfg.Expert.graph.drop_ratio,
                cfg=cfg,
                batch_norm=True,
                readout="mean")
            # expert projection layer
            self.expert_projection = layers.MLP(
                cfg.Expert.graph.feat_dim, cfg.Projection.expert_output_dim)

        elif self.cfg.train.Ablation.experiment_model == 'sequence':

            self.sequence_model = SequenceExpert(
                os.path.join(os.getenv("PRETRAINED_ROOT"),
                             cfg.Expert.sequence.model),
                cfg.Expert.sequence.tokenizer, cfg.Expert.sequence.padding,
                cfg.Expert.sequence.truncation, cfg.Expert.sequence.max_length,
                cfg.Expert.sequence.output_hidden_states)

            self.expert_projection = layers.MLP(
                cfg.Expert.sequence.output_dim,
                cfg.Projection.expert_output_dim)

        self.output_dim = cfg.Projection.expert_output_dim

    def forward(self, smiles, mol_descs, all_loss=None, metric=None):
        if isinstance(mol_descs, str):
            mol_descs = [mol_descs]
        if isinstance(smiles, str):
            smiles = [smiles]

        if self.cfg.train.Ablation.experiment_model == 'graph':
            expert_feature = self.graph_model(smiles)["graph_feature"]

        elif self.cfg.train.Ablation.experiment_model == 'sequence':
            expert_feature = self.sequence_model(
                smiles)["hidden_states"][-1][:, 0, :].squeeze(1)

        expert_embeding = self.expert_projection(expert_feature)

        return expert_embeding, [expert_feature]


class ExpertOnlyProphecy4Explain(Prophecy):

    def __init__(self, cfg):
        super().__init__(cfg)
        self.cfg = cfg

        # expert and llm
        self.graph_model = GraphExpert4Explain(
            pretrained_model_path=os.path.join(os.getenv("PRETRAINED_ROOT"),
                                               cfg.Expert.graph.model,
                                               "model.pth"),
            feat_dim=cfg.Expert.graph.feat_dim,
            emb_dim=cfg.Expert.graph.emb_dim,
            edge_feature_dim=None,
            num_graph_layers=cfg.Expert.graph.num_graph_layers,
            drop_ratio=cfg.Expert.graph.drop_ratio,
            batch_norm=True,
            readout="mean")
        # expert projection layer
        self.expert_projection = layers.MLP(cfg.Expert.graph.feat_dim,
                                            cfg.Projection.expert_output_dim)

        self.output_dim = cfg.Projection.expert_output_dim

    def forward(self, x, edge_index, edge_attr, batch):

        expert_feature = self.graph_model(x, edge_index, edge_attr,
                                          batch)["graph_feature"]

        expert_embeding = self.expert_projection(expert_feature)

        return expert_embeding, [expert_feature]

    def get_node_embedding(self, x):
        return self.graph_model.gnn.get_node_embedding(x)


class LLMOnlyProphecy4Explain(Prophecy):

    def __init__(self, cfg):
        super().__init__(cfg)
        self.cfg = cfg

        if cfg.train.Ablation.experiment_model == 'bert':
            self.llm = BertEmbedding(
                os.path.join(os.getenv("PRETRAINED_ROOT"),
                             cfg.LLMs.bert.model),
                os.path.join(os.getenv("PRETRAINED_ROOT"),
                             cfg.LLMs.bert.tokenizer), cfg.LLMs.bert.padding,
                cfg.LLMs.bert.truncation, cfg.LLMs.bert.max_length)
            # LLMs projection layer
            self.llm_projection = layers.MLP(cfg.LLMs.bert.output_dim,
                                             cfg.Projection.llm_output_dim)
        elif cfg.train.Ablation.experiment_model == 'llama':
            self.llm = LlamaEmbedding(
                    os.path.join(os.getenv("PRETRAINED_ROOT"),
                                cfg.LLMs.llama.model),
                    os.path.join(os.getenv("PRETRAINED_ROOT"),
                                cfg.LLMs.llama.tokenizer), cfg.LLMs.llama.padding,
                    cfg.LLMs.llama.truncation, cfg.LLMs.llama.max_length)

            self.llm_projection = layers.MLP(cfg.LLMs.llama.output_dim,
                                             cfg.Projection.llm_output_dim)
        if cfg.train.Ablation.experiment_model == 'bert':
            for param in self.llm.parameters():
                param.requires_grad = False
        elif cfg.train.Ablation.experiment_model == 'llama':
            if cfg.LLMs.llama.use_cache == False:
                for param in self.llm.parameters():
                    param.requires_grad = False

        self.output_dim = cfg.Projection.llm_output_dim

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
       
        text_output = self.llm.model(input_ids,attention_mask=attention_mask)
        if self.cfg.train.Ablation.experiment_model == 'bert':
            text_feature = text_output.last_hidden_state[:, 0, :].squeeze(1)
        else :
            text_feature = text_output.hidden_states[12][:,-1, :]

        text_embeding = self.llm_projection(text_feature)

        return text_embeding, [text_feature]