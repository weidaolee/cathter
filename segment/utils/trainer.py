import os
import copy
import json
import tqdm

import cv2
import torch
import numpy as np
import albumentations as alb

from termcolor import cprint

from torch.optim.lr_scheduler import CyclicLR
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from torch.utils.data import DataLoader

from utils.timer import TickTock
from utils.logger import TrainLogger, ValidLossLogger, ValidMetricLogger

from utils.postprocess import DilateSigmoidPostProcess
from utils.postprocess import DilateSoftmaxPostProcess
from utils.postprocess import AppearenceCorrectionPostProcess
from utils.postprocess import SigmoidPostProcess
from utils.postprocess import SoftmaxPostProcess

from tensorboardX import SummaryWriter


class Trainer:
    def __init__(
        self,
        args,
        cfg,
        loss,
        model,
        train_set,
        valid_set,
    ):
        print("Setting Arguments...", args)
        self.args = args
        self.cfg = cfg

        self.train_path = self.args.train_path
        self.valid_path = self.args.valid_path

        self.set_device()
        self.makedirs()
        self.load_checkpoint()
        self.train_configuring()

        self.prepare_model(model)
        self.prepare_optimizer()
        self.prepare_criterion(loss)
        self.prepare_datapipeline(train_set, valid_set)
        self.prepare_lr_scheduler()
        self.prepare_aug_scheduler()
        self.prepare_postprocessor()
        self.prepare_logger()

        self.initial_epoch()
        self.writer = SummaryWriter(self.log_path, flush_secs=1)

        self.evaluate()
        self.save_config()
        self.train()

    def set_device(self):
        # declare visible gpus
        os.environ["CUDA_VISIBLE_DEVICES"] = self.args.gpus
        print(f"Use GPU: {self.args.gpus}")

        # chose divice
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.device_ids = [int(i) for i in list(self.args.gpus.split(","))]
        # self.device = self.device_ids[0]
        # self.device_ids = list(range(len(list(self.args.gpus.split(",")))))

        self.workers = self.args.workers

    def makedirs(self):
        # make dirs
        self.result_base = f"./results/{self.args.task_name}"
        os.makedirs(self.result_base, exist_ok=True)

        self.save_path = os.path.join(self.result_base, "checkpoints")
        os.makedirs(self.save_path, exist_ok=True)

        self.log_path = os.path.join(self.result_base, "logs")
        os.makedirs(self.log_path, exist_ok=True)

    def load_checkpoint(self):
        if self.args.checkpoint is not None:
            print("Load checkpoint...", self.args.checkpoint)
            self.state = torch.load(self.args.checkpoint)
        else:
            print("No checkpoint will be loaded.")

    def train_configuring(self):
        # training config
        print("Training configuring...")
        print(self.cfg["train"])
        cfg = self.cfg["train"]
        self.batch_size = cfg["batch_size"]
        self.base_lr = cfg["learning_rate"]

        self.max_epoch = cfg["max_epoch"]
        self.weight_decay = cfg["weight_decay"]

    def prepare_model(self, model):
        print("Prepare model...")
        print("Model configuring...")
        cfg = self.cfg["model"]

        self.model = model

        self.model.encoder = torch.nn.DataParallel(
            self.model.encoder,
            device_ids=self.device_ids,
        )

        if hasattr(model, "decoder"):
            self.model.decoder = torch.nn.DataParallel(
                self.model.decoder,
                device_ids=self.device_ids,
            )

        self.model.classification_head = torch.nn.DataParallel(
            self.model.classification_head,
            device_ids=self.device_ids,
        )

        self.model = self.model.to(self.device, dtype=torch.float32)

        if cfg["load_ckpt"]:
            self.state["model"].pop("segmentation_head.0.weight")
            self.state["model"].pop("segmentation_head.0.bias")
            in_channels = cfg["parameters"]["in_channels"]
            w = self.state["model"]["encoder.module.conv1.0.weight"]

            if in_channels > w.shape[1]:
                print("Load model state...")
                w = self.state["model"].pop("encoder.module.conv1.0.weight")
                w = torch.cat([w[:, 0:1, :, :], w], 1)
                self.state["model"]["encoder.module.conv1.0.weight"] = w
            elif in_channels < w.shape[1]:
                self.state["model"].pop("encoder.module.conv1.0.weight")

        if cfg["load_ckpt"]:
            self.model.load_state_dict(self.state["model"], strict=False)
        else:
            print("No model state dict will be loaded.")

    def prepare_optimizer(self):
        print("Prepare optimizer...")

        self.optimizer = torch.optim.AdamW(
            params=self.model.parameters(),
            lr=self.base_lr,
            weight_decay=self.weight_decay,
            amsgrad=self.cfg["optimizer"]["amsgrad"],
        )
        print("optimizer: ", self.optimizer)

        if self.cfg["optimizer"]["load_ckpt"]:
            print("Load optimizer state...")
            self.optimizer.load_state_dict(self.state["optimizer"])
        else:
            print("No optimizer state dict will be loaded.")

    def prepare_criterion(self, loss):
        print("Prepare criterion...")
        self.criterion = loss

    def prepare_datapipeline(self, train_set, valid_set):
        self.train_set = train_set
        self.valid_set = valid_set

        # prepare dataloader
        print("Prepare dataloader...")
        self.train_loader = DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.workers,
            prefetch_factor=self.cfg["train"]["prefetch"],
            drop_last=True,
            # pin_memory=True,
            persistent_workers=True,
        )

        self.valid_loader = DataLoader(
            self.valid_set,
            batch_size=self.batch_size * 1,
            shuffle=False,
            num_workers=self.workers,
            prefetch_factor=self.cfg["train"]["prefetch"] * 2,
            persistent_workers=True,
        )

    @TickTock()
    def prepare_lr_scheduler(self):
        print("Prepare learning scheduler...")
        cfg = self.cfg["lr_scheduler"]

        if cfg["name"] == "CosineAnnealingWarmRestarts":
            lr_scheduler = CosineAnnealingWarmRestarts(
                self.optimizer,
                **cfg["parameters"],
            )
        elif cfg["name"] == "ReduceLROnPlateau":
            lr_scheduler = ReduceLROnPlateau(
                self.optimizer,
                **cfg["parameters"],
            )

        elif cfg["name"] == "CyclicLR":
            epoch_size_down = cfg["epoch_size_down"]
            step_size_down = int(len(self.train_loader) * epoch_size_down)

            lr_scheduler = CyclicLR(
                self.optimizer,
                max_lr=self.base_lr,
                step_size_down=step_size_down,
                **cfg["parameters"],
            )

        self.lr_scheduler = lr_scheduler

        if self.cfg["lr_scheduler"]["load_ckpt"]:
            print("Load scheduler state...")
            self.lr_scheduler.load_state_dict(self.state["lr_scheduler"])
        else:
            print("No scheduler state dict will be loaded.")

    def prepare_aug_scheduler(self):
        cfg = self.cfg["aug_scheduler"]
        max_epoch = self.cfg["train"]["max_epoch"]
        self.aug_scheduler = {}
        # self.aug_scheduler["p"] = np.clip(
        #     np.linspace(0.5, 0, 100)**(1 / 3), 0.5, 1)

        self.aug_scheduler["p"] = np.clip(np.linspace(0.75, 0.25, max_epoch),
                                          0.25, 0.5)

        for aug in cfg.keys():
            self.aug_scheduler[aug] = {
                k: np.linspace(cfg[aug][k][0], cfg[aug][k][1],
                               max_epoch).tolist()
                for k in cfg[aug].keys()
            }

    def prepare_postprocessor(self):
        cfg = self.cfg["postprocess"]

        if "DilateSigmoidPostProcess" in cfg:
            k = "DilateSigmoidPostProcess"
            self.postprocessor = DilateSigmoidPostProcess()

        elif "DilateSoftmaxPostProcess" in cfg:
            k = "DilateSoftmaxPostProcess"
            self.postprocessor = DilateSoftmaxPostProcess()

        if cfg[k]["scheduler"] is not None:
            lookup = cfg[k]["scheduler"]
            n = len(lookup)
            q = self.max_epoch // n
            r = self.max_epoch % n

            self.post_scheduler = []
            for i in range(n):
                for _ in range(q):
                    self.post_scheduler.append([1, lookup[i]])

            for _ in range(r):
                self.post_scheduler.append([1, lookup[-1]])

        else:
            self.post_scheduler = None

        k = "AppearenceCorrectionPostProcess"
        if k in cfg:
            self.postprocessor = AppearenceCorrectionPostProcess()
            self.post_scheduler = None

        k = "SigmoidPostProcess"
        if k in cfg:
            self.postprocessor = SigmoidPostProcess()
            self.post_scheduler = None

        k = "SoftmaxPostProcess"
        if k in cfg:
            self.postprocessor = SoftmaxPostProcess()
            self.post_scheduler = None

    def prepare_logger(self):
        print("Prepare logger...")
        self.train_logger = TrainLogger(trainer=self)

        self.valid_logger = {}
        self.valid_logger["loss"] = ValidLossLogger(trainer=self, key="loss")

        cfg = self.cfg["dataset"]
        for k in cfg["findings"]:
            self.valid_logger[k] = ValidMetricLogger(
                trainer=self,
                key=k,
                findings=cfg["findings"],
            )

    def initial_epoch(self):
        # initial epoch
        print("Initial epoch...")
        cfg = self.cfg["train"]

        self.global_step = 0
        self.current_epoch = 0

        self.max_step = self.max_epoch * len(self.train_loader)

        if cfg["monitor"]["mode"] == "min":
            self.best_metric = 1024
        else:
            self.best_metric = -1

        self.best_epoch = 0

        self.timer = TickTock(name="total", mode="return")
        self.timer.start()
        if cfg["continue"]:
            self.current_epoch = self.state["epoch"]
            print(f"Strat training from epoch: {self.current_epoch}")

    def schedule_aug(self, i):
        schedule_table = copy.deepcopy(self.aug_scheduler)

        p = self.aug_scheduler["p"][i]

        del schedule_table["p"]
        for aug in schedule_table:
            for params in schedule_table[aug]:
                schedule_table[aug][params] = schedule_table[aug][params][i]

        self.train_set.augmentations = alb.Compose([
            alb.RandomBrightnessContrast(
                **schedule_table["RandomBrightnessContrast"],
                brightness_by_max=True,
                p=p,
            ),
            alb.ShiftScaleRotate(
                **schedule_table["ShiftScaleRotate"],
                border_mode=cv2.BORDER_CONSTANT,
                p=p,
            ),
            alb.IAAPiecewiseAffine(
                scale=(0.01, schedule_table["IAAPiecewiseAffine"]["scale"]),
                nb_rows=5,
                nb_cols=5,
                p=p,
            ),
            alb.ElasticTransform(
                **schedule_table["ElasticTransform"],
                alpha_affine=0,
                border_mode=cv2.BORDER_CONSTANT,
                p=p,
            ),
            alb.GridDistortion(
                num_steps=10,
                **schedule_table["GridDistortion"],
                border_mode=cv2.BORDER_CONSTANT,
                p=p,
            ),
            alb.IAAPerspective(
                scale=(0., schedule_table["IAAPerspective"]["scale"]),
                p=p,
            ),
            alb.ChannelShuffle(p=1),
            alb.HorizontalFlip(**schedule_table["HorizontalFlip"]),
        ])

    def schedule_postprocess(self, i):
        if self.post_scheduler is not None:
            self.postprocessor.max_pool = self.post_scheduler[i]

    def train(self):
        print("Start training...")

        for i in range(self.max_epoch):
            self.schedule_aug(i)
            self.schedule_postprocess(i)

            self.current_epoch += 1
            self.train_an_epoch()
            self.evaluate()

            if isinstance(self.lr_scheduler, CosineAnnealingWarmRestarts):
                self.lr_scheduler.step()
            elif isinstance(self.lr_scheduler, ReduceLROnPlateau):
                self.lr_scheduler.step(self.best_metric)

        self.timer.stop()
        print("Finish training.")

    @TickTock(name="epoch")
    def train_an_epoch(self):
        ...

    @TickTock()
    def evaluate(self):
        ...

    def save_model(self):
        cfg = self.cfg["train"]["monitor"]
        epoch = self.current_epoch
        path = os.path.join(self.save_path, "model.pth")
        metric = cfg['metric']
        best_metric = self.best_metric

        save_dict = {
            "epoch": epoch,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict(),
        }
        cprint(f"save model at epoch {epoch} with {metric} {best_metric}...",
               color="red",
               attrs=["bold", "underline"])
        torch.save(save_dict, path)

    def save_config(self):
        save_path = os.path.join(self.result_base, "config.json")

        with open(save_path, "w") as f:
            json.dump(self.cfg, f, indent=4)


class SegmentTrainer(Trainer):
    def __init__(
        self,
        args,
        cfg,
        loss,
        model,
        train_set,
        valid_set,
    ):
        super(SegmentTrainer, self).__init__(
            args=args,
            cfg=cfg,
            loss=loss,
            model=model,
            train_set=train_set,
            valid_set=valid_set,
        )

    @TickTock(name="epoch")
    def train_an_epoch(self):
        self.model.train()
        for _, inputs, target in self.train_loader:
            # set grad to 0
            self.optimizer.zero_grad()

            # prepare input
            for k in inputs:
                inputs[k] = inputs[k].to(self.device)

            if "seg" in target:
                target["seg"] = target["seg"].to(self.device)

            for k in target["cls"]:
                target["cls"][k] = target["cls"][k].to(self.device)

            # inference
            output = self.model(inputs)

            # post-process
            output, target = self.postprocessor(output, target)

            # compute loss
            loss = self.criterion(output, target)
            del output, target

            # backward
            loss.backward()
            self.optimizer.step()

            if isinstance(self.lr_scheduler, CyclicLR):
                self.lr_scheduler.step()

            # log
            self.global_step += 1
            self.train_logger.update()

    @TickTock()
    def evaluate(self):
        print("Start evaluating...")
        with torch.no_grad():
            self.model.eval()

            num_batchs = len(self.valid_loader)
            for _, inputs, target in tqdm.tqdm(self.valid_loader,
                                               total=num_batchs):

                # prepare input
                for k in inputs:
                    inputs[k] = inputs[k].to(self.device)

                if "seg" in target:
                    target["seg"] = target["seg"].to(self.device)

                for k in target["cls"]:
                    target["cls"][k] = target["cls"][k].to(self.device)

                # inference
                output = self.model(inputs)

                # post-process
                self.postprocessor.max_pool = [1, 7]
                output, target = self.postprocessor(output, target)

                # compute loss
                _ = self.criterion(output, target)

                for k in self.valid_logger:
                    output = copy.deepcopy(output)
                    target = copy.deepcopy(target)

                    self.valid_logger[k].update(output, target)

                del output, target

            for k in self.valid_logger:
                self.valid_logger[k].evaluate()


class ClassifyTrainer(Trainer):
    def __init__(
        self,
        args,
        cfg,
        loss,
        model,
        train_set,
        valid_set,
    ):
        super(ClassifyTrainer, self).__init__(
            args=args,
            cfg=cfg,
            loss=loss,
            model=model,
            train_set=train_set,
            valid_set=valid_set,
        )

    @TickTock(name="epoch")
    def train_an_epoch(self):
        self.model.train()
        for _, inputs, target in self.train_loader:
            # set grad to 0
            self.optimizer.zero_grad()

            # prepare input
            for k in inputs:
                inputs[k] = inputs[k].to(self.device)

            for k in target["cls"]:
                target["cls"][k] = target["cls"][k].to(self.device)

            # inference
            output = self.model(inputs)

            # post-process
            output, target = self.postprocessor(output, target)

            # compute loss
            loss = self.criterion(output, target)
            del output, target

            # backward
            loss.backward()
            self.optimizer.step()

            if isinstance(self.lr_scheduler, CyclicLR):
                self.lr_scheduler.step()

            # log
            self.global_step += 1
            self.train_logger.update()

    @TickTock()
    def evaluate(self):
        print("Start evaluating...")
        with torch.no_grad():
            self.model.eval()

            num_batchs = len(self.valid_loader)
            for _, inputs, target in tqdm.tqdm(self.valid_loader,
                                               total=num_batchs):
                # prepare input
                for k in inputs:
                    inputs[k] = inputs[k].to(self.device)

                for k in target["cls"]:
                    target["cls"][k] = target["cls"][k].to(self.device)

                # inference
                output = self.model(inputs)

                # post-process
                output, target = self.postprocessor(output, target)

                # compute loss
                _ = self.criterion(output, target)

                for k in self.valid_logger:
                    output = copy.deepcopy(output)
                    target = copy.deepcopy(target)

                    self.valid_logger[k].update(output, target)

                del output, target

            for k in self.valid_logger:
                self.valid_logger[k].evaluate()
