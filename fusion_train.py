import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import nni
import torch
import random
import numpy as np
import pandas as pd


os.environ["TORCH_VERSION"] = torch.__version__

from mol_prophecy.mytorchdrug.MyTaskEngine import MyTaskEngine
from mol_prophecy.mytorchdrug.datasets.MyClintTox import MyClinTox
from mol_prophecy.mytorchdrug.datasets.MyBACE import MyBACE
from mol_prophecy.mytorchdrug.datasets.MyFreeSolv import MyFreeSolv
from mol_prophecy.mytorchdrug.datasets.MySIDER import MySIDER
from mol_prophecy.mytorchdrug.MyPropertyPrediction import MyPropertyPrediction as fusionTask
from mol_prophecy.ablation_experiment.AblationProphecy import LLMOnlyProphecy, ExpertOnlyProphecy
from omegaconf import OmegaConf
from mol_prophecy.model.prophecy import Prophecy
from mol_prophecy.utils.nni_utils import update_cfg
from mol_prophecy.utils.scheduler import lr_scheduler
from torchdrug import data,tasks


def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


os.environ['CUDA_VISIBLE_DEVICES'] = "1"
if __name__ == "__main__":
    cfg = OmegaConf.load("./config/config.yaml")

    # 此处为我们需要调优的所有超参数 可以设置一个初始值
    if cfg.mode.nni:
        cfg = update_cfg(cfg)

    set_seed(cfg.train.random_seed)

    # -------------------- Define the model --------------------
    if cfg.train.Ablation.is_ablation == False:
        mol_prophecy = Prophecy(cfg)
    elif cfg.train.Ablation.is_ablation == True:# ablation experiment
        if cfg.train.Ablation.experiment_model == 'bert' or cfg.train.Ablation.experiment_model == 'llama':
            mol_prophecy = LLMOnlyProphecy(cfg)
        elif cfg.train.Ablation.experiment_model == 'graph' or cfg.train.Ablation.experiment_model == 'sequence':
            mol_prophecy = ExpertOnlyProphecy(cfg)

    # -------------------- Define the task --------------------
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

    lengths = [int(0.8 * len(dataset)), int(0.1 * len(dataset))]
    lengths += [len(dataset) - sum(lengths)]

    if cfg.train.data_split == "random":
        train_set, valid_set, test_set = torch.utils.data.random_split(
            dataset, lengths)
    elif cfg.train.data_split == "scaffold":
                    
        # torch.manual_seed(8)  # 数据集固定随机种子
        train_set, valid_set, test_set = data.scaffold_split(dataset, lengths)
        # print(f"Seed {8}: Train set size = {len(train_set)}")

        # test_smiles = []
        # test_label = []
        # for sample in test_set:
        #     test_smiles.append(sample['smiles'])
        #     test_label.append(sample['Class'])
        # test_df = pd.DataFrame({'smiles':test_smiles,'label':test_label})
        # test_df.to_csv(f'./test_{8}.csv',index=False)
    
    # set random seed for reproducibility 又把随机种子设置回来 
    # torch.manual_seed(cfg.train.random_seed)

    # -------------------- Define the task --------------------
    if cfg.train.dataset in ['bace', 'clintox','sider']:
        criterion = "bce"
        metric = ("auprc", "auroc")
        default_metric = "auroc"
    elif cfg.train.dataset in ['freesolv']:
        criterion = "mse"
        metric = ("mae", "rmse")
        default_metric = "rmse"

    task = fusionTask(model=mol_prophecy,
                      task=dataset.tasks,
                      criterion=criterion,
                      metric=metric,
                      default_metric=default_metric,
                      num_mlp_layer=cfg.Head.num_mlp_layer,
                      mlp_dropout=cfg.Head.dropout)

    # -------------------- Load the pretrained graph model --------------------

    # -------------------- Train the model --------------------

    optimizer = torch.optim.Adam(task.parameters(),
                                 lr=cfg.train.learning_rate,
                                 betas=(0.9, 0.999),
                                 eps=cfg.train.adam_epsilon,
                                 weight_decay=cfg.train.weight_decay)
    scheduler = lr_scheduler(cfg, optimizer)

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
    solver.train(num_epoch=cfg.train.num_epoch)

    test_metric = solver.evaluate("test")
    print('use_handcrafted_features: ',cfg.Handcrafted.use_handcrafted_features,'use_fp_handcrafted_features: ',cfg.FP_Handcrafted.use_fp_handcrafted_features )
    for i_mt in solver.model.metric:
        if i_mt == 'auroc' or i_mt == 'rmse':
            if i_mt in tasks._metric_name:
                real_name =  tasks._metric_name[i_mt]
            else:
                real_name = i_mt
            metric_score = []
            for i_task in solver.model.task:
                metric_score.append(test_metric["%s [%s]" % (real_name, i_task)])
            metric_score = torch.stack(metric_score)
            model_mean_metric = (metric_score * solver.model.weight).sum() / solver.model.weight.sum()
            model_mean_metric = round(model_mean_metric.cpu().item(), 4)
            
            print(f"Test/{i_mt}: {model_mean_metric}")
    
    if cfg.mode.nni:
        nni.report_final_result(model_mean_metric)
