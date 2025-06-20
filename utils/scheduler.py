#!/usr/bin/env python3
# -*- coding:utf-8 -*-

from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau, LambdaLR, CyclicLR,StepLR,MultiStepLR

def lr_scheduler(cfg, optimizer):

    scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 1)

    if cfg.train.lr_scheduler.type == "cosine":
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=cfg.train.lr_scheduler.cosine.T_0,
            T_mult=cfg.train.lr_scheduler.cosine.T_mult,
            eta_min=cfg.train.lr_scheduler.cosine.eta_min, verbose=False)
    elif cfg.train.lr_scheduler.type == "cycle":

        def func(x):
            return cfg.train.lr_scheduler.cycle.max_lr_decrease_factor**(x - 1)

        scheduler = CyclicLR(optimizer=optimizer,
                                base_lr=cfg.train.lr_scheduler.cycle.base_lr,
                                max_lr=cfg.train.lr_scheduler.cycle.max_lr,
                                step_size_up=cfg.train.lr_scheduler.cycle.step_size_up,
                                step_size_down=cfg.train.lr_scheduler.cycle.step_size_down,
                                scale_fn=func,
                                cycle_momentum = False,
                                gamma=cfg.train.lr_scheduler.cycle.gamma, verbose=False)


    elif cfg.train.lr_scheduler.type == "plateau":
        if cfg.train.dataset in ['bace', 'clintox','sider']: #classification
            mode = "max"
        elif cfg.train.dataset in ['freesolv']: #regression
            mode = "min"

        scheduler = ReduceLROnPlateau(optimizer,
                                      mode=mode,
                                      factor=cfg.train.lr_scheduler.plateau.factor,
                                      patience=cfg.train.lr_scheduler.plateau.patience,
                                      threshold=0.0001,
                                      threshold_mode='rel',
                                      cooldown=0,
                                      min_lr=cfg.train.lr_scheduler.plateau.min_lr,
                                      eps=1e-08,
                                      verbose=False)
    elif cfg.train.lr_scheduler.type == "StepLR":
        scheduler = StepLR(optimizer, step_size=cfg.train.lr_scheduler.StepLR.step_size, gamma=cfg.train.lr_scheduler.StepLR.gamma)
    elif cfg.train.lr_scheduler.type == "MultiStepLR":
        scheduler = MultiStepLR(optimizer, milestones=cfg.train.lr_scheduler.MultiStepLR.milestones, gamma=cfg.train.lr_scheduler.MultiStepLR.gamma)

    return scheduler