import os
import json
import datetime
import tqdm

import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

from utils.logger import TrainLogger
from utils.lr_scheduler import GradualWarmupSchedulerV2

from tensorboardX import SummaryWriter


class Trainer:
    def __init__(
        self,
        args,
        model,
        dataset,
        criterion,
        valid_logger,
        load_ckpt,
    ):
        print("Setting Arguments...", args)
        self.args = args

        self.train_path = self.args.train_path
        self.valid_path = self.args.valid_path

        # load config
        with open("./config.json", 'r') as f:
            self.cfg = json.load(f)
            print("Configuring...", self.cfg)

        self.model = model
        self.criterion = criterion
        self.dataset = dataset

        self.set_device()
        self.makedirs()
        self.training_configuring()
        self.prepare_datapipeline()

        if load_ckpt:
            if args.checkpoint is None:
                raise IOError("please define a checkpoint path.")
        self.load_ckpt = load_ckpt
        self.initial_epoch()

        self.writer = SummaryWriter(self.log_path, flush_secs=1)
        self.train_logger = TrainLogger(self)
        self.valid_logger = valid_logger(self)

        self.evaluate()

    def set_device(self):
        # declare visible gpus
        # os.environ["CUDA_VISIBLE_DEVICES"] = self.args.gpu
        print(f"Use GPU: {self.args.gpu}")

        # chose divice
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        self.workers = self.args.workers

    def makedirs(self):
        # make dirs
        self.result_base = f"./results/{self.args.prefix}"
        os.makedirs(self.result_base, exist_ok=True)

        self.save_path = os.path.join(self.result_base, "checkpoints")
        os.makedirs(self.save_path, exist_ok=True)

        self.log_path = os.path.join(self.result_base, "logs")
        os.makedirs(self.log_path, exist_ok=True)

    def training_configuring(self):
        # training config
        self.batch_size = self.cfg["batch_size"]
        self.prefetch = self.cfg["prefetch"]
        self.base_lr = self.cfg["learning_rate"]
        # self.lr_schedule = self.cfg["schedule"]
        self.warmup = self.cfg["warmup"]
        self.max_epoch = self.cfg["max_epoch"]
        self.weight_decay = self.cfg["weight_decay"]

        # define model
        print("Prepare model...")
        self.prepare_model()

        # define loss
        print("loss: ", self.criterion)

        # define optimizer
        self.optimizer = torch.optim.Adam(
            params=self.model.parameters(),
            lr=self.base_lr * self.batch_size / self.warmup,
            weight_decay=self.weight_decay,
        )
        print("optimizer: ", self.optimizer)

    def prepare_model(self):

        # load checkpoint
        # if self.load_ckpt:
        #     print("loading pytorch ckpt...", self.args.checkpoint)

        #     state = torch.load(self.args.checkpoint)
        #     if 'model_state_dict' in state.keys():
        #         self.model.load_state_dict(state['model_state_dict'])
        #     else:
        #         self.model.load_state_dict(state)
        self.model.summary()
        self.model = self.model.to(self.device)
        self.model = torch.nn.DataParallel(self.model)

    def prepare_datapipeline(self):
        # prepare dataset
        print("Prepare dataset...")
        self.train_set = self.dataset(self.train_path, is_train=True)
        self.valid_set = self.dataset(self.valid_path, is_train=False)

        # prepare dataloader
        print("Prepare dataloader...")
        self.train_loader = DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.workers,
            prefetch_factor=self.prefetch,
            drop_last=True,
            # pin_memory=True,
            # persistent_workers=True,
        )

        self.valid_loader = DataLoader(
            self.valid_set,
            batch_size=self.batch_size * 2,
            shuffle=False,
            num_workers=self.workers,
            # pin_memory=True,
            # persistent_workers=True,
        )

    def initial_epoch(self):
        # initial epoch

        print("Initial epoch...")
        self.global_step = 0
        self.current_epoch = 0

        self.max_step = self.max_epoch * len(self.train_loader)
        self.scheduler_cosine = CosineAnnealingLR(
            self.optimizer,
            T_max=self.max_epoch - 1,
            eta_min=1e-8,
        )
        self.scheduler = GradualWarmupSchedulerV2(
            self.optimizer,
            multiplier=10,
            total_epoch=1,
            after_scheduler=self.scheduler_cosine,
        )
        self.best_auc = -1
        self.best_epoch = 0

        self.start_time = datetime.datetime.now()

    def train(self):
        print("Start training...")
        for i in range(self.max_epoch):
            self.current_epoch += 1

            self.train_an_epoch()
            self.scheduler.step()
            self.evaluate()

    def train_an_epoch(self):
        self.model.train()
        for image, target in self.train_loader:
            # to divice
            image = image.to(self.device)
            target = target.to(self.device)

            # set grad to 0
            self.optimizer.zero_grad()

            # inference
            output = self.model(image)

            # compute loss
            loss = self.criterion(output, target)

            # backward
            loss.backward()
            self.optimizer.step()

            # log
            self.global_step += 1
            self.train_logger.update()

    def evaluate(self):
        print("Start evaluating...")
        with torch.no_grad():
            self.model.eval()
            for image, target in tqdm.tqdm(self.valid_loader,
                                           total=len(self.valid_loader)):
                # to divice
                image.to(self.device)
                target.to(self.device)

                # inference
                output = self.model(image)

                # compute loss
                _ = self.criterion(output.cuda(), target.cuda())

                self.valid_logger.update(output, target)
            self.valid_logger.evaluate()

    def save_model(self):
        epoch = self.current_epoch
        path = os.path.join(self.save_path, f"model_{epoch:03d}.pth")

        best_auc = self.best_auc

        save_dict = {
            "epoch": epoch,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
        }
        print(f"save model at epoch {epoch:03d} with AUC {best_auc}...")
        torch.save(save_dict, path)
