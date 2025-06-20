#!/usr/bin/env python3
# -*- coding:utf-8 -*-


import nni

def update_cfg(cfg):
    # get trialID
    trial_id = nni.get_trial_id()
    # initialize the params
    optimized_params = nni.get_next_parameter()
    if not optimized_params == {}:
        # update the config before training
        
        cfg.train.lr_scheduler.plateau.factor = optimized_params["plateau_factor"]
        cfg.train.learning_rate = float(optimized_params["learning_rate"])
        cfg.train.lr_scheduler.plateau.patience = optimized_params["plateau_patience"]
        cfg.Head.dropout = float(optimized_params["head_dropout"])
        cfg.train.GatedXattnDenseLayer.dropout = float(optimized_params["attention_dropout"])
        cfg.train.early_stop_patience = optimized_params["early_stop_patience"]
        
    return cfg
