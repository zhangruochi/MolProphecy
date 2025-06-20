import os
import sys
import logging
from itertools import islice
import numpy as np
import torch
from torch import distributed as dist
from torch import nn
from torch.utils import data as torch_data
import re
from torchdrug import data, core, utils, tasks
from torchdrug.core import Registry as R
from torchdrug.utils import comm, pretty
import time
from torch.utils.tensorboard import SummaryWriter  # Import for logging
import shutil
module = sys.modules[__name__]
logger = logging.getLogger(__name__)


@R.register("core.MyTaskEngine")
class MyTaskEngine(core.Engine):

    def __init__(self,
                 task,
                 train_set,
                 valid_set,
                 test_set,
                 optimizer,
                 cfg,
                 scheduler=None,
                 gpus=None,
                 batch_size=1,
                 early_stop_patience=5,
                 gradient_interval=1,
                 num_worker=0,
                 logger="logging",
                 log_interval=100):

        self.rank = comm.get_rank()
        self.world_size = comm.get_world_size()
        self.gpus = gpus
        self.batch_size = batch_size
        self.gradient_interval = cfg.train.gradient_interval
        self.num_worker = num_worker
        self.early_stop_patience = early_stop_patience
        self.cfg = cfg
        self.best_model_path = ""

        if gpus is None:
            self.device = torch.device("cpu")
        else:
            if len(gpus) != self.world_size:
                error_msg = "World size is %d but found %d GPUs in the argument"
                if self.world_size == 1:
                    error_msg += ". Did you launch with `python -m torch.distributed.launch`?"
                raise ValueError(error_msg % (self.world_size, len(gpus)))
            self.device = torch.device(gpus[self.rank % len(gpus)])

        if self.world_size > 1 and not dist.is_initialized():
            if self.rank == 0:
                module.logger.info("Initializing distributed process group")
            backend = "gloo" if gpus is None else "nccl"
            comm.init_process_group(backend, init_method="env://")

        if hasattr(task, "preprocess"):
            if self.rank == 0:
                module.logger.warning("Preprocess training set")
            # TODO: more elegant implementation
            # handle dynamic parameters in optimizer
            old_params = list(task.parameters())
            result = task.preprocess(train_set, valid_set, test_set)
            if result is not None:
                train_set, valid_set, test_set = result
            new_params = list(task.parameters())
            if len(new_params) != len(old_params):
                optimizer.add_param_group(
                    {"params": new_params[len(old_params):]})
                min_lr = len(optimizer.param_groups) * [float(cfg.train.lr_scheduler.plateau.min_lr)]
                scheduler.min_lrs = min_lr
        if self.world_size > 1:
            task = nn.SyncBatchNorm.convert_sync_batchnorm(task)
            buffers_to_ignore = []
            for name, buffer in task.named_buffers():
                if not isinstance(buffer, torch.Tensor):
                    buffers_to_ignore.append(name)
            task._ddp_params_and_buffers_to_ignore = set(buffers_to_ignore)
        if self.device.type == "cuda":
            task = task.cuda(self.device)

        self.model = task
        if self.model.default_metric == "auprc" or self.model.default_metric == "auroc":
            self.best_metric = 0
        elif self.model.default_metric == "rmse":
            self.best_metric = float('inf')

        self.train_set = train_set
        self.valid_set = valid_set
        self.test_set = test_set
        self.optimizer = optimizer
        self.scheduler = scheduler

        if isinstance(logger, str):
            if logger == "logging":
                logger = core.LoggingLogger()
            elif logger == "wandb":
                logger = core.WandbLogger(project=task.__class__.__name__)
            else:
                raise ValueError("Unknown logger `%s`" % logger)
        self.meter = core.Meter(log_interval=log_interval,
                                silent=self.rank > 0,
                                logger=logger)
        self.meter.log_config(self.config_dict())

    def train(self, num_epoch=1, batch_per_epoch=None):
        """
        Train the model.

        If ``batch_per_epoch`` is specified, randomly draw a subset of the training set for each epoch.
        Otherwise, the whole training set is used for each epoch.

        Parameters:
            num_epoch (int, optional): number of epochs
            batch_per_epoch (int, optional): number of batches per epoch
        """
        sampler = torch_data.DistributedSampler(self.train_set,
                                                self.world_size, self.rank)
        dataloader = data.DataLoader(self.train_set,
                                     self.batch_size,
                                     sampler=sampler,
                                     num_workers=self.num_worker)
        batch_per_epoch = batch_per_epoch or len(dataloader)
        model = self.model
        model.split = "train"
        if self.world_size > 1:
            if self.device.type == "cuda":
                model = nn.parallel.DistributedDataParallel(
                    model,
                    device_ids=[self.device],
                    find_unused_parameters=True)
            else:
                model = nn.parallel.DistributedDataParallel(
                    model, find_unused_parameters=True)

        if self.cfg.train.Ablation.is_ablation == False:
            summary_comment = "_{}_{}_{}_{}".format(
                self.cfg.train.dataset, self.cfg.train.fusion_approach,
                self.cfg.train.data_split, self.cfg.train.lr_scheduler.type)
            summary_comment += "_{}+{}".format(self.cfg.Expert.model,
                                               self.cfg.LLMs.model)
        elif self.cfg.train.Ablation.is_ablation == True:
            summary_comment = "_{}_ablation_{}_{}".format(
                self.cfg.train.dataset,
                self.cfg.train.data_split, self.cfg.train.lr_scheduler.type)
            summary_comment += "_{}".format(self.cfg.train.Ablation.experiment_model)

        writer = SummaryWriter(
            comment=summary_comment+'_{}'.format(self.cfg.train.random_seed))  # Create a SummaryWriter for logging
        model_save_time = time.strftime("%Y%m%d-%H%M%S")

        patience = self.early_stop_patience  # Set the patience for early stopping
        best_loss = float('inf')  # Initialize best loss
        no_improvement_count = 0  # Counter for no improvement

        for epoch in self.meter(num_epoch):
            model.train()
            sampler.set_epoch(epoch)
            metrics = []
            start_id = 0
            # the last gradient update may contain less than gradient_interval batches
            gradient_interval = min(batch_per_epoch - start_id,
                                    self.gradient_interval)

            for batch_id, batch in enumerate(
                    islice(dataloader, batch_per_epoch)):

                if self.device.type == "cuda":
                    batch = utils.cuda(batch, device=self.device)

                loss, metric = model(batch)

                if not loss.requires_grad:
                    raise RuntimeError(
                        "Loss doesn't require grad. Did you define any loss in the task?"
                    )
                loss = loss / gradient_interval
                loss.backward()
                metrics.append(metric)

                if batch_id - start_id + 1 == gradient_interval:
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                    metric = utils.stack(metrics, dim=0)
                    # Update to handle multiple metrics:
                    for key, value in metric.items():
                        value = utils.mean(value, dim=0)
                        if self.world_size > 1:
                            value = comm.reduce(value, op="mean")
                        writer.add_scalar(
                            f"Loss/{key}", value,
                            self.meter.epoch2batch[-1] + batch_id)
                        self.meter.update({key: value})

                    metrics = []
                    start_id = batch_id + 1
                    gradient_interval = min(batch_per_epoch - start_id,
                                            self.gradient_interval)
                if self.cfg.train.lr_scheduler_use == True:
                    if self.cfg.train.lr_scheduler.when == "batch" and self.cfg.train.lr_scheduler.type in (
                            "cosine", "cycle", "StepLR", "MultiStepLR"):
                        self.scheduler.step()

                # log lr for each batch

                if self.rank == 0:
                    writer.add_scalar("lr",
                                      self.optimizer.param_groups[0]["lr"],
                                      self.meter.epoch2batch[-1] + batch_id)

            # Check for early stopping after each epoch as well:
            index = slice(self.meter.epoch2batch[-1],
                          self.meter.epoch2batch[-1] + self.meter.batch_id)
            record = []
            for k, v in self.meter.records.items():
                record.append(np.mean(v[index]))

            train_metric = self.evaluate("train")
            for i_mt in self.model.metric:
                if i_mt in tasks._metric_name:
                    real_name =  tasks._metric_name[i_mt]
                else:
                    real_name = i_mt
                metric_score = []
                for i_task in self.model.task:
                    metric_score.append(train_metric["%s [%s]" % (real_name, i_task)])
                    if len(self.model.task) > 1:
                        writer.add_scalar(f"Train/{real_name} [{i_task}]", train_metric["%s [%s]" % (real_name, i_task)], epoch)
                average_score = torch.stack(metric_score).mean()
                writer.add_scalar(f"Train/{i_mt}", average_score, epoch)

            valid_metric = self.evaluate("valid")
            for i_mt in self.model.metric:
                if i_mt in tasks._metric_name:
                    real_name =  tasks._metric_name[i_mt]
                else:
                    real_name = i_mt
                metric_score = []
                for i_task in self.model.task:
                    metric_score.append(valid_metric["%s [%s]" % (real_name, i_task)])
                    if len(self.model.task) > 1:
                        writer.add_scalar(f"Valid/{real_name} [{i_task}]", valid_metric["%s [%s]" % (real_name, i_task)], epoch)
                average_score = torch.stack(metric_score).mean()
                writer.add_scalar(f"Valid/{i_mt}", average_score, epoch)
                print(f"Valid/{i_mt}: {average_score}")

                # current date and time for saving the model
                if self.cfg.train.Ablation.is_ablation == False:
                    save_model_path = os.path.join(
                        os.path.dirname(os.path.dirname(
                            os.path.abspath(__file__))),
                        "./saved_model/{}_{}_{}/{}_{}_{}_{}_{}_{}_{}_epoch{}.pth".
                        format(self.cfg.train.dataset,self.cfg.train.random_seed, model_save_time,
                            self.cfg.train.dataset,
                            self.cfg.train.fusion_approach,
                            self.cfg.train.data_split, self.cfg.LLMs.model,
                            self.cfg.train.lr_scheduler.type, i_mt, average_score,
                            epoch))
                elif self.cfg.train.Ablation.is_ablation == True:
                    save_model_path = os.path.join(
                        os.path.dirname(os.path.dirname(
                            os.path.abspath(__file__))),
                        "./saved_model/{}_ablation_{}_{}_{}/{}_{}_{}_{}_{}_{}_{}_epoch{}.pth".
                        format(self.cfg.train.dataset,self.cfg.train.Ablation.experiment_model,self.cfg.train.random_seed, model_save_time,
                            self.cfg.train.dataset,
                            self.cfg.train.fusion_approach,
                            self.cfg.train.data_split, self.cfg.train.Ablation.experiment_model,
                            self.cfg.train.lr_scheduler.type, i_mt, average_score,
                            epoch))

                if not os.path.exists(os.path.dirname(save_model_path)):
                    os.makedirs(os.path.dirname(save_model_path))

                if i_mt == 'auroc':
                    if average_score > self.best_metric:
                        # Delete previous best model if exists
                        if os.path.exists(self.best_model_path):
                            os.remove(self.best_model_path)
                        self.best_model_path = save_model_path
                        self.best_metric = average_score
                        print(f'Best model epoch is {epoch} and best auroc is {self.best_metric}')
                        # Save the new best model immediately
                        self.save(self.best_model_path)
                        no_improvement_count = 0
                    else:
                        no_improvement_count += 1   
                        print(f'No improvement for {no_improvement_count} epochs')


                if i_mt == 'rmse':
                    if average_score < self.best_metric:
                        # Delete previous best model if exists
                        if os.path.exists(self.best_model_path):
                            os.remove(self.best_model_path)
                        self.best_model_path = save_model_path
                        self.best_metric = average_score
                        print(f'Best model epoch is {epoch} and best rmse is {self.best_metric}')  
                        # Save the new best model immediately
                        self.save(self.best_model_path)
                        no_improvement_count = 0
                    else:
                        no_improvement_count += 1
                        print(f'No improvement for {no_improvement_count} epochs')

            # bace sider use auroc as early stopping metric  freesolv use rmse as early stopping metric
            #early stopping
            if no_improvement_count >= patience:
                print(f'Early stopping triggered! Best model epoch is {epoch-patience}')
                break  # Exit the training loop
            
            if self.cfg.train.lr_scheduler_use == True:
                if self.cfg.train.lr_scheduler.when == "epoch" and self.cfg.train.lr_scheduler.type in (
                        "cosine", "cycle", "StepLR", "MultiStepLR"):
                    self.scheduler.step()

                if self.cfg.train.lr_scheduler.when == "epoch" and self.cfg.train.lr_scheduler.type == "plateau":
                    default_metric_name = tasks._get_metric_name(self.model.default_metric)
                    
                    matching_metrics = [metric_v for metric_k, metric_v in valid_metric.items() if default_metric_name in metric_k]
                    valid_average_score = torch.stack(matching_metrics).mean()
                    self.scheduler.step(valid_average_score)
        checkpoint = torch.load(self.best_model_path)["model"]
        module.logger.info("Loading the best model from {}".format(
            self.best_model_path))
        self.model.load_state_dict(checkpoint, strict=False)
        self.model.eval()
        test_metric = self.evaluate("test")
        for i_mt in self.model.metric:
            if i_mt in tasks._metric_name:
                real_name =  tasks._metric_name[i_mt]
            else:
                real_name = i_mt
            metric_score = []
            for i_task in self.model.task:
                metric_score.append(test_metric["%s [%s]" % (real_name, i_task)])
                if len(self.model.task) > 1:
                    writer.add_scalar(f"Test/{real_name} [{i_task}]", test_metric["%s [%s]" % (real_name, i_task)], epoch)
            average_score = torch.stack(metric_score).mean()
            writer.add_scalar(f"Test/{i_mt}", average_score, epoch)
            print(f"Test/{i_mt}: {average_score}")

        writer.close()  # Close the SummaryWriter
        
        #获取self.best_model_path的路径
        destination_file = os.path.dirname(self.best_model_path)
        source_file = './config/config.yaml'
        shutil.copy(source_file, os.path.join(destination_file, 'config.yaml'))
