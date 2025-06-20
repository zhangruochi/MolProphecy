import os
import pandas as pd
import math
import torch
from torchdrug import tasks, layers
from collections import defaultdict
from torchdrug.core import Registry as R
from torchdrug.layers import functional
from torch.nn import functional as F

@R.register("tasks.MyPropertyPrediction")
class MyPropertyPrediction(tasks.PropertyPrediction):
    eps = 1e-10
    _option_members = {"task", "criterion", "metric"}

    def __init__(self,
                 model,
                 task=(),
                 criterion="mse",
                 metric=("mae", "rmse"),
                 default_metric="rmse",
                 num_mlp_layer=1,
                 normalization=True,
                 num_class=None,
                 mlp_batch_norm=False,
                 mlp_dropout=0,
                 graph_construction_model=None,
                 verbose=0):
        super().__init__(model=model,
                         task=task,
                         criterion=criterion,
                         metric=metric,
                         num_mlp_layer=num_mlp_layer,
                         normalization=normalization,
                         num_class=num_class,
                         mlp_batch_norm=mlp_batch_norm,
                         mlp_dropout=mlp_dropout,
                         graph_construction_model=graph_construction_model,
                         verbose=verbose)
        self.model = model
        self.task = task
        self.criterion = criterion
        self.metric = metric
        self.default_metric = default_metric
        self.num_mlp_layer = num_mlp_layer
        # For classification tasks, we disable normalization tricks.
        self.normalization = normalization and ("ce" not in criterion) and (
            "bce" not in criterion)
        self.num_class = (num_class, ) if isinstance(num_class,
                                                     int) else num_class
        self.mlp_batch_norm = mlp_batch_norm
        self.mlp_dropout = mlp_dropout
        self.graph_construction_model = graph_construction_model
        self.verbose = verbose

        # load_dotenv()
        # openai.api_key = os.getenv("OPENAI_API_KEY")
        # self.cfg = OmegaConf.load("config/config.yaml")

    def preprocess(self, train_set, valid_set, test_set):
        """
        Compute the mean and derivation for each task on the training set.
        """
        # calculate the number of positive and negative samples for each task
        task_positive_counts = defaultdict(int)
        task_negative_counts = defaultdict(int)
        values = defaultdict(list)
        for sample in train_set:
            if not sample.get("labeled", True):
                continue
            for task in self.task:
                if not math.isnan(sample[task]):
                    values[task].append(sample[task])
                    if sample[task] == 1:
                        task_positive_counts[task] += 1
                    else:
                        task_negative_counts[task] += 1
        
        # Register pos_weight for each task
        for task in self.task:
            pos = task_positive_counts[task]
            neg = task_negative_counts[task]
            if pos > 0:
                pos_weight = torch.tensor(neg / pos, dtype=torch.float32)
            else:
                pos_weight = torch.tensor(1.0)
            self.register_buffer(f"class_pos_weight_{task}", pos_weight)
        
        mean = []
        std = []
        weight = []
        num_class = []

        for task, w in self.task.items():
            value = torch.tensor(values[task])
            mean.append(value.float().mean())
            std.append(value.float().std())
            weight.append(w)
            if value.ndim > 1:
                num_class.append(value.shape[1])
            elif value.dtype == torch.long:
                task_class = value.max().item()
                if task_class == 1 and "bce" in self.criterion:
                    num_class.append(1)
                else:

                    num_class.append(task_class + 1)
            else:
                num_class.append(1)

        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float))
        self.register_buffer("std", torch.as_tensor(std, dtype=torch.float))
        self.register_buffer("weight",
                             torch.as_tensor(weight, dtype=torch.float))
        self.num_class = self.num_class or num_class

        hidden_dims = [self.model.output_dim] * (self.num_mlp_layer - 1)
        self.mlp = layers.MLP(self.model.output_dim,
                              hidden_dims + [sum(self.num_class)],
                              batch_norm=self.mlp_batch_norm,
                              dropout=self.mlp_dropout)

    def predict(self, batch, all_loss=None, metric=None):

        text = batch["text"]
        smiles = batch["smiles"]

        embedding,_ = self.model(smiles, text)
        pred = self.mlp(embedding)

        if self.normalization:
            pred = pred * self.std + self.mean

        return pred
    
    def forward(self, batch):
        """"""
        all_loss = torch.tensor(0, dtype=torch.float32, device=self.device)
        metric = {}

        pred = self.predict(batch, all_loss, metric)

        if all([t not in batch for t in self.task]):
            # unlabeled data
            return all_loss, metric

        target = self.target(batch)
        labeled = ~torch.isnan(target)
        target[~labeled] = 0

        for criterion, weight in self.criterion.items():
            if criterion == "mse":
                if self.normalization:
                    loss = F.mse_loss((pred - self.mean) / self.std, (target - self.mean) / self.std, reduction="none")
                else:
                    loss = F.mse_loss(pred, target, reduction="none")
            elif criterion == "bce":
                loss_list = []
                for i, task in enumerate(self.task):
                    task_pred = pred[:, i]
                    task_target = target[:, i]
                    task_labeled = labeled[:, i]

                    if hasattr(self, f"class_pos_weight_{task}"):
                        pos_weight = getattr(self, f"class_pos_weight_{task}")
                    else:
                        pos_weight = torch.tensor(1.0, device=pred.device)

                    task_loss = self.focal_bce_with_logits(
                        task_pred, task_target, gamma=2.0, pos_weight=pos_weight, reduction="none"
                    )
                    task_loss = functional.masked_mean(task_loss, task_labeled, dim=0)
                    loss_list.append(task_loss)

                loss = torch.stack(loss_list, dim=0)
            elif criterion == "ce":
                loss = F.cross_entropy(pred, target.long().squeeze(-1), reduction="none").unsqueeze(-1)
            else:
                raise ValueError("Unknown criterion `%s`" % criterion)
            loss = functional.masked_mean(loss, labeled, dim=0)

            name = tasks._get_criterion_name(criterion)
            if self.verbose > 0:
                for t, l in zip(self.task, loss):
                    metric["%s [%s]" % (name, t)] = l
            loss = (loss * self.weight).sum() / self.weight.sum()
            metric[name] = loss
            all_loss += loss * weight

        return all_loss, metric
    
    def focal_bce_with_logits(self, pred, target, gamma=2.0, pos_weight=None, reduction="none"):
        """
        pred: raw logits (N,)
        target: binary target (N,)
        pos_weight: scalar tensor, for imbalanced classes
        """
        bce_loss = F.binary_cross_entropy_with_logits(pred, target, pos_weight=pos_weight, reduction="none")
        
        # Convert logits to probabilities
        prob = torch.sigmoid(pred)
        p_t = prob * target + (1 - prob) * (1 - target)  # p_t = p if y==1 else 1-p

        focal_term = (1 - p_t) ** gamma
        loss = focal_term * bce_loss

        if reduction == "mean":
            return loss.mean()
        elif reduction == "sum":
            return loss.sum()
        return loss
