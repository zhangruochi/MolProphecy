#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: /home/richard/projects/mol_prophecy/model/multimodality.py
# Project: /home/richard/projects/mol_prophecy/model
# Created Date: Thursday, January 25th 2024, 8:54:17 pm
# Author: Ruochi Zhang
# Email: zrc720@gmail.com
# -----
# Last Modified: Fri Jun 20 2025
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
import json
import numpy as np
import torch
from torch import nn
from torchdrug import layers
from torch.nn.parameter import Parameter
import pickle
from .mols.graph_expert import GraphExpert
from .llms.llm_bert import BertEmbedding,LlamaEmbedding
from .seqs.sequence_expert import SequenceExpert
from .utils import cal_physicochemical_properties
from .gatedxattndenselayer import GatedXattnDenseLayer
from .handcraftattention import HandcraftedAttentionFusion

# load .env file
from dotenv import load_dotenv

load_dotenv()


class Prophecy(nn.Module):

    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg

        # expert and llm
        if cfg.Expert.model == 'graph':
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

        elif cfg.Expert.model == 'sequence':

            self.sequence_model = SequenceExpert(
                os.path.join(os.getenv("PRETRAINED_ROOT"),
                             cfg.Expert.sequence.model),
                cfg.Expert.sequence.tokenizer, cfg.Expert.sequence.padding,
                cfg.Expert.sequence.truncation, cfg.Expert.sequence.max_length,
                cfg.Expert.sequence.output_hidden_states)

        if cfg.LLMs.model == 'bert':
            self.llm = BertEmbedding(
                os.path.join(os.getenv("PRETRAINED_ROOT"),
                             cfg.LLMs.bert.model),
                os.path.join(os.getenv("PRETRAINED_ROOT"),
                             cfg.LLMs.bert.tokenizer), cfg.LLMs.bert.padding,
                cfg.LLMs.bert.truncation, cfg.LLMs.bert.max_length)
        elif cfg.LLMs.model == 'llama':
            # if gpu memory is not enough, use cache
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

        # LLMs projection layer
        if self.cfg.LLMs.model == 'bert':
            self.llm_projection = nn.Sequential(
                                        nn.Linear(cfg.LLMs.bert.output_dim, cfg.Projection.llm_output_dim),
                                        nn.ReLU(),
                                        nn.Dropout(0.3)  # add dropout
                                    )
        elif self.cfg.LLMs.model == 'llama':
            self.llm_projection = nn.Sequential(
                                        nn.Linear(cfg.LLMs.llama.output_dim, cfg.Projection.llm_output_dim),
                                        nn.ReLU(),
                                        nn.Dropout(0.1) 
                                    )

        # add handcrafted features
        # load cache pkl file
        if self.cfg.Handcrafted.use_handcrafted_features == True:
            with open(os.path.join("./data/{}/smiles2handcraft.json".format(self.cfg.train.dataset)), 'r') as f:
                self.handcrafted_cache_dict = json.load(f)
            self.handcrafted_projection = nn.Sequential(
                                        nn.Linear(cfg.Handcrafted.feature_dim, cfg.Projection.fusion_output_dim),
                                        nn.ReLU(),  
                                        nn.Dropout(0.3) 
                                    )
        if self.cfg.FP_Handcrafted.use_fp_handcrafted_features == True:
            with open(os.path.join("./data/{}/smiles2fphandcraft.json".format(self.cfg.train.dataset)), 'r') as f:
                self.fp_handcrafted_cache_dict = json.load(f)
            self.fp_handcrafted_projection = nn.Sequential(
                                        nn.Linear(cfg.FP_Handcrafted.feature_dim, cfg.Projection.fusion_output_dim),
                                        nn.ReLU(),  
                                        nn.Dropout(0.3) 
                                    )
        
        # expert projection layer
        if self.cfg.Expert.model == 'graph':
            self.expert_projection = nn.Sequential(
                                        nn.Linear(cfg.Expert.graph.feat_dim, cfg.Projection.expert_output_dim),
                                        nn.ReLU(),
                                        nn.Dropout(0.1) 
                                    )
        elif self.cfg.Expert.model == 'sequence':
            self.expert_projection = nn.Sequential(
                                        nn.Linear(cfg.Expert.sequence.output_dim, cfg.Projection.expert_output_dim),
                                        nn.ReLU(),
                                        nn.Dropout(0.3)
                                    )
            
        #fusion MLP layer
        self.fusion_projection = nn.Sequential(
                                                nn.Linear(cfg.Projection.expert_output_dim,cfg.Projection.fusion_output_dim),
                                                nn.ReLU(),
                                                nn.Dropout(cfg.Head.dropout)  
                                            )
        
        

        if self.cfg.train.fusion_approach == 'concat':
            self.output_dim = cfg.Projection.expert_output_dim + cfg.Projection.llm_output_dim
        elif self.cfg.train.fusion_approach == 'tensor_fusion':
            self.output_dim = (cfg.Projection.expert_output_dim +
                               1) * (cfg.Projection.llm_output_dim + 1)
        elif self.cfg.train.fusion_approach == 'bilinear_fusion':
            self.output_dim = (cfg.Projection.expert_output_dim) * (
                cfg.Projection.llm_output_dim)
        elif self.cfg.train.fusion_approach == 'low_rank_fusion':
            self.output_dim = 64
        elif self.cfg.train.fusion_approach == 'attention':
            if self.cfg.Handcrafted.use_handcrafted_features == True:
                self.handcrafted_attn_fusion = HandcraftedAttentionFusion(
                    embed_dim=cfg.Projection.fusion_output_dim,
                    num_heads=4,
                    num_layers=cfg.train.GatedXattnDenseLayer.num_layers,
                    dropout=0.3
                )
                self.output_dim = cfg.Projection.fusion_output_dim
            elif self.cfg.FP_Handcrafted.use_fp_handcrafted_features == True:
                self.fp_handcrafted_attn_fusion = HandcraftedAttentionFusion(
                    embed_dim=cfg.Projection.fusion_output_dim,
                    num_heads=4,
                    num_layers=cfg.train.GatedXattnDenseLayer.num_layers,
                    dropout=0.3
                )

                self.output_dim = cfg.Projection.fusion_output_dim
            else:
                self.output_dim = cfg.Projection.expert_output_dim
            
            
            self.attn_layers = nn.ModuleList([GatedXattnDenseLayer(cfg.Projection.expert_output_dim, cfg.train.GatedXattnDenseLayer.num_head,dropout=cfg.train.GatedXattnDenseLayer.dropout) for _ in range(cfg.train.GatedXattnDenseLayer.num_layers)])

        # freeze the expert and llm model
        if cfg.LLMs.freeze:
            if cfg.LLMs.model == 'bert':
                for param in self.llm.parameters():
                    param.requires_grad = False
            elif cfg.LLMs.model == 'llama':
                if cfg.LLMs.llama.use_cache == False:
                    for param in self.llm.parameters():
                        param.requires_grad = False

        if cfg.Expert.freeze:
            for param in self.graph_model.parameters():
                param.requires_grad = False

        # physicochemical properties
        if len(self.cfg.Expert.physicochemical_properties) > 0:
            self.physicochemical_properties = self.cfg.Expert.physicochemical_properties

            self.phy_projection = layers.MLP(
                len(self.cfg.Expert.physicochemical_properties),
                cfg.Projection.physicochemical_properties_output_dim)
            self.output_dim += cfg.Projection.physicochemical_properties_output_dim

    def forward(self, smiles, mol_descs, all_loss=None, metric=None):

        if isinstance(mol_descs, str):
            mol_descs = [mol_descs]
        if isinstance(smiles, str):
            smiles = [smiles]

        if self.cfg.Expert.model == 'graph':
            graph_output = self.graph_model(smiles)
            expert_feature = graph_output["graph_feature"]
            expert_mask = graph_output["graph_mask"]
            h_analysis = graph_output["h_analysis"]
        elif self.cfg.Expert.model == 'sequence':
            if self.cfg.train.fusion_approach == 'attention':
                expert_feature = self.sequence_model(
                    smiles)["hidden_states"][-1]
            else:
                expert_feature = self.sequence_model(
                    smiles)["hidden_states"][-1][:, 0, :].squeeze(1)

        if self.cfg.LLMs.model == 'bert':
            text_output,text_mask = self.llm(mol_descs)
            if self.cfg.train.fusion_approach == 'attention':
                # attention input shape is (batch_size,sequence_length,embedding_dim)
                text_feature = text_output.last_hidden_state
            else:
                text_feature = text_output.last_hidden_state[:, 0, :].squeeze(1)
            # this is a variable used for model analysis
            text_feature_analysis = text_output.last_hidden_state[:, 0, :]
        elif self.cfg.LLMs.model == 'llama':
            if self.cfg.LLMs.llama.use_cache == False:
                text_output,text_mask = self.llm(mol_descs)
                embedding = text_output.hidden_states[12]
                if self.cfg.train.fusion_approach == 'attention':
                    text_feature = embedding
                else:
                    text_feature = embedding[:,-1, :]
                
                # this is a variable used for model analysis
                text_feature_analysis = embedding[:,-1,:]

            else:
                #load cache npy file and attention mask file
                embeddings = []
                text_mask = []
                text_feature_analysis_list = []
                for smiles_i in smiles:
                    file_path = self.llama_cache_dict.get(smiles_i, None)
                    embedding = np.load(file_path)
                    mask_file_path = self.llama_cache_mask_dict.get(smiles_i, None)
                    mask_attn = np.load(mask_file_path)
                    if self.cfg.train.fusion_approach == 'attention':
                        embeddings.append(embedding)
                        text_mask.append(mask_attn)
                    else:
                        embeddings.append(embedding[:,-1, :])
                    text_feature_analysis_list.append(embedding[:,-1,:])
                text_feature = torch.tensor(np.concatenate(embeddings, axis=0),device=self.device)
                text_feature_analysis = torch.tensor(np.concatenate(text_feature_analysis_list, axis=0),device=self.device)
                if self.cfg.train.fusion_approach == 'attention':
                    text_mask = torch.tensor(np.concatenate(text_mask, axis=0),device=self.device)

        # add handcrafted physicochemical properties
        if self.cfg.Handcrafted.use_handcrafted_features == True:
            handcrafted_embedding_list = []
            for smiles_i in smiles:
                handcrafted_pickle_path = self.handcrafted_cache_dict.get(smiles_i, None)
                with open(handcrafted_pickle_path, 'rb') as f:
                    handcrafted_embedding = pickle.load(f)
                handcrafted_embedding_list.append(handcrafted_embedding)
            handcrafted_feature_analysis = torch.tensor(np.array(handcrafted_embedding_list),device=self.device)
        
        #add fp handcrafted features
        if self.cfg.FP_Handcrafted.use_fp_handcrafted_features == True:
            fp_handcrafted_embedding_list = []
            for smiles_i in smiles:
                fp_handcrafted_pickle_path = self.fp_handcrafted_cache_dict.get(smiles_i, None)
                with open(fp_handcrafted_pickle_path, 'rb') as f:
                    fp_handcrafted_embedding = pickle.load(f)
                fp_handcrafted_embedding_list.append(fp_handcrafted_embedding)
            fp_handcrafted_feature_analysis = torch.tensor(np.array(fp_handcrafted_embedding_list),device=self.device)
            

        
        text_embeding = self.llm_projection(text_feature)
        expert_embeding = self.expert_projection(expert_feature)

        if self.cfg.train.fusion_approach == 'concat':
            fusion_embedding = torch.cat([expert_embeding, text_embeding],
                                         dim=1)
        elif self.cfg.train.fusion_approach == 'tensor_fusion':
            fusion_embedding = tensor_fusion(expert_embeding, text_embeding)
        elif self.cfg.train.fusion_approach == 'bilinear_fusion':
            fusion_embedding = bilinear_fusion(expert_embeding, text_embeding)
        elif self.cfg.train.fusion_approach == 'low_rank_fusion':
            fusion_embedding = lowRank_fusion(expert_embeding, text_embeding)
        elif self.cfg.train.fusion_approach == 'attention':
            fusion_embedding = expert_embeding  # 初始是 expert_embeding
            for layer in self.attn_layers:
                #expert_embeding and text_embeding are both (batch_size, sequence_length, embedding_dim)
                fusion_embedding = layer(fusion_embedding, text_embeding,expert_mask,text_mask)
            #pooling
            fusion_embedding = fusion_embedding.mean(dim=1)
            fusion_embedding = self.fusion_projection(fusion_embedding)

        if len(self.cfg.Expert.physicochemical_properties) > 0:
            features = self.cfg.Expert.physicochemical_properties
            feature_array = []
            for smi in smiles:
                feature_array.append(
                    cal_physicochemical_properties(smi, features))
            feature_array = torch.tensor(feature_array,
                                         dtype=torch.float32).to(self.device)

            phy_features = self.phy_projection(feature_array)

            fusion_embedding = torch.cat([fusion_embedding, phy_features],
                                     dim=1)
        if self.cfg.Handcrafted.use_handcrafted_features == True:
            #convert to float32
            handcrafted_feature_analysis = handcrafted_feature_analysis.float()
            handcrafted_feature_analysis = self.handcrafted_projection(handcrafted_feature_analysis)
            fusion_embedding = self.handcrafted_attn_fusion(fusion_embedding, handcrafted_feature_analysis)
        elif self.cfg.FP_Handcrafted.use_fp_handcrafted_features == True:
            #convert to float32
            fp_handcrafted_feature_analysis = fp_handcrafted_feature_analysis.float()
            fp_handcrafted_feature_analysis = self.fp_handcrafted_projection(fp_handcrafted_feature_analysis)   
            fusion_embedding = self.fp_handcrafted_attn_fusion(fusion_embedding, fp_handcrafted_feature_analysis)
        
        return fusion_embedding, [text_feature_analysis, h_analysis]

    @property
    def device(self):
        return next(self.parameters()).device


########################################################################
def convert_list_to_str(input_list):
    result_list = []

    for item in input_list:
        if isinstance(item, float) or isinstance(item, int):
            result_list.append(str(item))
        elif isinstance(item, list):
            result_list.append(''.join(map(str, item)))
        else:
            result_list.append(str(item))

    return result_list


# tensor fusion for 2 modalities
def bilinear_fusion(v_D, v_P):
    A = v_D.unsqueeze(2)  # [n, A, 1]
    B = v_P.unsqueeze(1)  # [n, 1, B]
    fusion_AB = torch.einsum('nxt, nty->nxy', A, B)  # [n, A, B]
    fusion_AB = fusion_AB.flatten(start_dim=1)  #.unsqueeze(1) # [n, AxB, 1]
    return fusion_AB


def tensor_fusion(v_D, v_P):
    n = v_D.shape[0]
    device = v_D.device
    A = torch.cat([v_D, torch.ones(n, 1).to(device)], dim=1)
    B = torch.cat([v_P, torch.ones(n, 1).to(device)], dim=1)
    A = A.unsqueeze(2)  # [n, A, 1]
    B = B.unsqueeze(1)  # [n, 1, B]
    fusion_AB = torch.einsum('nxt, nty->nxy', A, B)  # [n, A, B]
    fusion_AB = fusion_AB.flatten(start_dim=1)  #.unsqueeze(1) # [n, AxB, 1]
    return fusion_AB


# low rank fusion for 2 modality
def lowRank_fusion(v_D, v_P, dim=64, rank=2):
    n = v_D.shape[0]
    device = v_D.device
    A = torch.cat([v_D, torch.ones(n, 1).to(device)], dim=1)
    B = torch.cat([v_P, torch.ones(n, 1).to(device)], dim=1)
    # set rank and the expected fusion dimension
    R = rank
    h = dim
    Wa = Parameter(torch.ones(R, A.shape[1], h).to(device))
    Wb = Parameter(torch.ones(R, B.shape[1], h).to(device))
    Wf = Parameter(torch.ones(1, R).to(device))
    bias = Parameter(torch.zeros(1, h).to(device))

    # decomposite modality A and B
    fusion_A = torch.matmul(A, Wa)
    fusion_B = torch.matmul(B, Wb)

    # modality Fusion
    funsion_AB = fusion_A * fusion_B
    funsion_AB = torch.matmul(Wf, funsion_AB.permute(1, 0, 2)).squeeze() + bias
    #funsion_AB = torch.concat([funsion_AB, v_D, v_P],1)
    return funsion_AB
