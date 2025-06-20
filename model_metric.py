import os
import sys
# sys.path.append("./torchdrug")
import torch
from torchdrug import data,tasks
from mytorchdrug.MyTaskEngine import MyTaskEngine
from mytorchdrug.datasets.MyClintTox import MyClinTox
from mytorchdrug.datasets.MyBACE import MyBACE
from mytorchdrug.datasets.MyFreeSolv import MyFreeSolv
from mytorchdrug.datasets.MySIDER import MySIDER
from mytorchdrug.MyPropertyPrediction import MyPropertyPrediction as fusionTask
from utils.scheduler import lr_scheduler
from omegaconf import OmegaConf
from model.prophecy import Prophecy
from ablation_experiment.AblationProphecy import LLMOnlyProphecy, ExpertOnlyProphecy
from torchdrug import data
import torch
import random
import scipy.stats
from torch.nn import functional as F
from torchdrug import utils
import numpy as np
from utils.model_evl_utils import classify_task_metric, regression_task_metric
import warnings
from torch.utils import data as torch_data
from matplotlib import pyplot as plt

warnings.filterwarnings('ignore')


def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False



if __name__ == "__main__":

    cfg = OmegaConf.load("./config/config.yaml")
    set_seed(cfg.train.random_seed)
    checkpoint_path = './saved_model/bace.pth'
    if cfg.train.Ablation.is_ablation == False:
        mol_prophecy = Prophecy(cfg)
    elif cfg.train.Ablation.is_ablation == True:
        if cfg.train.Ablation.experiment_model == 'bert' or cfg.train.Ablation.experiment_model == 'llama':
            mol_prophecy = LLMOnlyProphecy(cfg)
        elif cfg.train.Ablation.experiment_model == 'graph' or cfg.train.Ablation.experiment_model == 'sequence':
            mol_prophecy = ExpertOnlyProphecy(cfg)

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

    if cfg.train.dataset in ['bace', 'clintox','sider']:
        classify_task = True
        criterion = "bce"
        metric = ("auprc", "auroc")
        default_metric = "auroc"
    elif cfg.train.dataset in ['freesolv']:
        classify_task = False
        criterion = "mse"
        metric = ("mae", "rmse")
        default_metric = "rmse"

    lengths = [int(0.8 * len(dataset)), int(0.1 * len(dataset))]
    lengths += [len(dataset) - sum(lengths)]

    if cfg.train.data_split == "random":
        train_set, valid_set, test_set = torch.utils.data.random_split(
            dataset, lengths)
    elif cfg.train.data_split == "scaffold":
        train_set, valid_set, test_set = data.scaffold_split(
            dataset, lengths)
    smiles_list = [s["smiles"] for s in test_set]
    
    task = fusionTask(model=mol_prophecy,
                      task=dataset.tasks,
                      criterion=criterion,
                      metric=metric,
                      default_metric=default_metric,
                      num_mlp_layer=cfg.Head.num_mlp_layer,
                      mlp_dropout=cfg.Head.dropout)

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

    
    checkpoint = torch.load(checkpoint_path)["model"]
    task.load_state_dict(checkpoint, strict=False)
    solver.model.eval()
    sampler = torch_data.DistributedSampler(test_set, solver.world_size, solver.rank)
    dataloader = data.DataLoader(test_set, solver.batch_size, sampler=sampler, num_workers=solver.num_worker)
    preds = []
    targets = []
    for batch in dataloader:
        if solver.device.type == "cuda":
            batch = utils.cuda(batch, device=solver.device)
        with torch.no_grad():
            pred, target = solver.model.predict_and_target(batch)
        preds.append(pred)
        targets.append(target)

    pred = utils.cat(preds)
    target = utils.cat(targets)
    if classify_task:

        predicted_probabilities = F.sigmoid(pred)
        classify_task_metric(predicted_probabilities, target, cfg,
                             dataset.tasks, test_set.indices, smiles_list)
        
        test_metric = solver.evaluate("test")
        for i_mt in solver.model.metric:
            if i_mt in tasks._metric_name:
                real_name =  tasks._metric_name[i_mt]
            else:
                real_name = i_mt
            metric_score = []
            for i_task in solver.model.task:
                metric_score.append(test_metric["%s [%s]" % (real_name, i_task)])
            metric_score = torch.stack(metric_score)
            average_score = (metric_score * solver.model.weight).sum() / solver.model.weight.sum()
            print(f"Test/{i_mt}: {average_score}")

    else:
        test_metric = solver.evaluate("test")

        pred_np = pred.cpu().numpy().ravel()
        target_np = target.cpu().numpy().ravel()

        spearman_corr, _ = scipy.stats.spearmanr(pred_np, target_np)
        print(f"Spearman correlation: {spearman_corr}")

        pearson_corr, _ = scipy.stats.pearsonr(pred_np, target_np)
        print(f"Pearson correlation: {pearson_corr}")

        metric_save_root_path = './tmp/{}/scatter.png'.format(cfg.train.dataset)
        plt.figure(figsize=(8, 6))
        plt.scatter(pred_np, target_np, alpha=0.5)
        plt.xlabel('Predictions')
        plt.ylabel('Targets')
        plt.title('Scatter plot of Predictions vs Targets')
        plt.grid(True)
        plt.savefig(metric_save_root_path)
        plt.show()



    
