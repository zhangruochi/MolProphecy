#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: /home/richard/projects/mol_prophecy/inference.py
# Project: /home/richard/projects/mol_prophecy
# Created Date: Wednesday, January 31st 2024, 12:09:30 pm
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
import sys
import torch
import random
import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from torchdrug import data, tasks, utils
from torch.nn import functional as F
from torch.utils import data as torch_data

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mol_prophecy.mytorchdrug.MyTaskEngine import MyTaskEngine
from mol_prophecy.mytorchdrug.datasets.MyClintTox import MyClinTox
from mol_prophecy.mytorchdrug.datasets.MyBACE import MyBACE
from mol_prophecy.mytorchdrug.datasets.MyFreeSolv import MyFreeSolv
from mol_prophecy.mytorchdrug.datasets.MySIDER import MySIDER
from mol_prophecy.mytorchdrug.MyPropertyPrediction import MyPropertyPrediction as fusionTask
from mol_prophecy.ablation_experiment.AblationProphecy import LLMOnlyProphecy, ExpertOnlyProphecy
from mol_prophecy.model.prophecy import Prophecy
from mol_prophecy.utils.scheduler import lr_scheduler


def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


if __name__ == "__main__":

    # Set checkpoint path - modify this path to your trained model
    checkpoint_path = './saved_model/<your_checkpoint_path>'

    # Load configuration
    cfg = OmegaConf.load("./config/config.yaml")
    set_seed(cfg.train.random_seed)
    
    # Check if checkpoint exists
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found at {checkpoint_path}")
        print("Please train the model first or specify the correct checkpoint path")
        exit(1)
    
    # Define the model
    if cfg.train.Ablation.is_ablation == False:
        mol_prophecy = Prophecy(cfg)
    elif cfg.train.Ablation.is_ablation == True:
        if cfg.train.Ablation.experiment_model == 'bert' or cfg.train.Ablation.experiment_model == 'llama':
            mol_prophecy = LLMOnlyProphecy(cfg)
        elif cfg.train.Ablation.experiment_model == 'graph' or cfg.train.Ablation.experiment_model == 'sequence':
            mol_prophecy = ExpertOnlyProphecy(cfg)
    
    # Load dataset
    if cfg.train.dataset == 'clintox':
        dataset = MyClinTox("~/molecule-datasets/",
                            cfg.chemist.chatgpt_oracle.clintox.path)
    elif cfg.train.dataset == 'bace':
        dataset = MyBACE("~/molecule-datasets/",
                         cfg.chemist.chatgpt_oracle.bace.path)
    elif cfg.train.dataset == 'freesolv':
        dataset = MyFreeSolv("~/molecule-datasets/",
                             cfg.chemist.chatgpt_oracle.freesolv.path)
    elif cfg.train.dataset == 'sider':
        dataset = MySIDER("~/molecule-datasets/",
                          cfg.chemist.chatgpt_oracle.sider.path)
    
    # Determine task type
    if cfg.train.dataset in ['bace', 'clintox', 'sider']:
        classify_task = True
        criterion = "bce"
        metric = ("auprc", "auroc")
        default_metric = "auroc"
    elif cfg.train.dataset in ['freesolv']:
        classify_task = False
        criterion = "mse"
        metric = ("mae", "rmse")
        default_metric = "rmse"
    
    # Split dataset
    lengths = [int(0.8 * len(dataset)), int(0.1 * len(dataset))]
    lengths += [len(dataset) - sum(lengths)]
    
    if cfg.train.data_split == "random":
        train_set, valid_set, test_set = torch.utils.data.random_split(
            dataset, lengths)
    elif cfg.train.data_split == "scaffold":
        train_set, valid_set, test_set = data.scaffold_split(
            dataset, lengths)
    
    # Get SMILES list for test set
    smiles_list = [s["smiles"] for s in test_set]
    
    # Define the task
    task = fusionTask(model=mol_prophecy,
                      task=dataset.tasks,
                      criterion=criterion,
                      metric=metric,
                      default_metric=default_metric,
                      num_mlp_layer=cfg.Head.num_mlp_layer,
                      mlp_dropout=cfg.Head.dropout)
    
    # Setup optimizer and scheduler (needed for model loading)
    optimizer = torch.optim.Adam(task.parameters(),
                                 lr=cfg.train.learning_rate,
                                 betas=(0.9, 0.999),
                                 eps=cfg.train.adam_epsilon,
                                 weight_decay=cfg.train.weight_decay)
    scheduler = lr_scheduler(cfg, optimizer)
    
    # Setup solver
    solver = MyTaskEngine(task,
                          train_set,
                          valid_set,
                          test_set,
                          optimizer,
                          cfg,
                          scheduler,
                          gpus=cfg.train.device_ids,
                          batch_size=cfg.train.batch_size,
                          early_stop_patience=cfg.train.early_stop_patience)
    
    # Load trained model
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if "model" in checkpoint:
        task.load_state_dict(checkpoint["model"], strict=False)
    else:
        task.load_state_dict(checkpoint, strict=False)
    
    # Set model to evaluation mode
    solver.model.eval()
    
    # Create data loader for test set
    sampler = torch_data.DistributedSampler(test_set, solver.world_size, solver.rank)
    dataloader = data.DataLoader(test_set, solver.batch_size, sampler=sampler, num_workers=solver.num_worker, collate_fn=data.graph_collate)
    
    # Run inference
    preds = []
    targets = []
    smiles_results = []
    
    print("Running inference...")
    for batch_idx, batch in enumerate(dataloader):
        if solver.device.type == "cuda":
            batch = utils.cuda(batch, device=solver.device)
        
        with torch.no_grad():
            pred, target = solver.model.predict_and_target(batch)
        
        preds.append(pred)
        targets.append(target)
        
        # Get SMILES for this batch - batch is now a collated dictionary
        batch_smiles = batch["smiles"]
        smiles_results.extend(batch_smiles)
        
        if batch_idx % 10 == 0:
            print(f"Processed {batch_idx + 1} batches...")
    
    # Concatenate results
    pred = utils.cat(preds)
    target = utils.cat(targets)
    
    # Process predictions based on task type
    if classify_task:
        predicted_probabilities = F.sigmoid(pred)
        predictions = predicted_probabilities.cpu().numpy()
    else:
        predictions = pred.cpu().numpy()
    
    # Create results DataFrame
    results_df = pd.DataFrame({
        'smiles': smiles_results,
        'prediction': predictions.flatten(),
        'target': target.cpu().numpy().flatten()
    })
    
    # Save results
    output_dir = f"./result/{cfg.train.dataset}"
    os.makedirs(output_dir, exist_ok=True)
    output_path = f"{output_dir}/inference_results.csv"
    results_df.to_csv(output_path, index=False)
    
    print(f"Inference completed. Results saved to {output_path}")
    
    # Print evaluation metrics
    if classify_task:
        test_metric = solver.evaluate("test")
        for i_mt in solver.model.metric:
            if i_mt in tasks._metric_name:
                real_name = tasks._metric_name[i_mt]
            else:
                real_name = i_mt
            metric_score = []
            for i_task in solver.model.task:
                metric_score.append(test_metric["%s [%s]" % (real_name, i_task)])
            metric_score = torch.stack(metric_score)
            average_score = (metric_score * solver.model.weight).sum() / solver.model.weight.sum()
            print(f"Test/{i_mt}: {average_score}")
    else:
        # For regression tasks, calculate correlation
        pred_np = pred.cpu().numpy().ravel()
        target_np = target.cpu().numpy().ravel()
        
        from scipy import stats
        spearman_corr, _ = stats.spearmanr(pred_np, target_np)
        pearson_corr, _ = stats.pearsonr(pred_np, target_np)
        
        print(f"Spearman correlation: {spearman_corr:.4f}")
        print(f"Pearson correlation: {pearson_corr:.4f}")
        
        # Calculate RMSE and MAE
        rmse = np.sqrt(np.mean((pred_np - target_np) ** 2))
        mae = np.mean(np.abs(pred_np - target_np))
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")