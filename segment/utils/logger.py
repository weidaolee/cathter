import os
import time
import warnings

import torch
import numpy as np

from itertools import cycle
from termcolor import cprint, colored

from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score


class TrainLogger:
    def __init__(self, trainer):
        self.trainer = trainer

    def update(self):
        warnings.simplefilter("ignore")
        step = self.trainer.global_step

        cfg = self.trainer.cfg["train"]
        trainer = self.trainer
        writer = trainer.writer

        max_step = trainer.max_step
        epoch = trainer.current_epoch
        max_epoch = trainer.max_epoch

        if "_last_lr" not in trainer.lr_scheduler.state_dict():
            lr = trainer.base_lr * trainer.batch_size
        else:
            lr = trainer.lr_scheduler.state_dict()["_last_lr"][0]

        loss_dict = self.trainer.criterion.loss_dict.copy()

        spent_time = trainer.timer.cumulative_time()
        best_metric = trainer.best_metric
        best_epoch = trainer.best_epoch

        epoch = step / len(self.trainer.train_loader)
        colors = cycle(["green", "blue", "magenta", "cyan"])
        attrs = ["bold"]

        cprint(
            f"epoch {epoch:.2f} {max_epoch}",
            end=colored("||", "grey"),
            color=next(colors),
            attrs=attrs,
        )

        cprint(
            f"step {step} {max_step}",
            end=colored("||", "grey"),
            color=next(colors),
            attrs=attrs,
        )

        cprint(
            f"time {spent_time}",
            end=colored("||", "grey"),
            color=next(colors),
            attrs=attrs,
        )

        cprint(
            f"lr {lr:.2e}",
            end=colored("||", "grey"),
            color=next(colors),
            attrs=attrs,
        )

        for loss in cfg["running_losses"]:
            value = loss_dict[loss]
            cprint(
                f"{loss} {value:.4f}",
                end=colored("||", "grey"),
                color=next(colors),
                attrs=attrs,
            )

        metric = cfg['monitor']['metric']
        cprint(
            f"best {metric} {best_metric:.4f} at {best_epoch}",
            end=colored("||\n", "grey"),
            color="red",
            attrs=attrs,
        )

        for k in loss_dict:
            writer.add_scalar(f"running/loss/{k}", loss_dict[k], step,
                              time.time())

        writer.add_scalar("lr", lr, step)


class Metric:
    def __init__(self, logger):
        self.key = logger.key
        self.logger = logger
        self.metric_dict = logger.metric_dict

        self.map = {
            f"{self.key}_auc": self.auc,
            f"{self.key}_f1": self.f1,
            f"{self.key}_precision": self.precision,
            f"{self.key}_recall": self.recall,
            f"{self.key}_acc": self.acc,
        }

    def compute_metrics(self, y_true, y_pred, y_prob, category):
        self.y_true = y_true
        self.y_pred = y_pred
        self.y_prob = y_prob
        self.category = c = category

        for m in self.metric_dict:
            self.metric_dict[m][c] = self.map[m]()

    def auc(self):
        return roc_auc_score(self.y_true, self.y_prob)

    def f1(self):
        return f1_score(self.y_true, self.y_pred)

    def precision(self):
        return precision_score(self.y_true, self.y_pred)

    def recall(self):
        return recall_score(self.y_true, self.y_pred)

    def acc(self):
        return accuracy_score(self.y_true, self.y_pred)


class ValidMetricLogger:
    def __init__(self,
                 trainer,
                 key,
                 findings,
                 metric_list=[
                     "auc",
                     "f1",
                     "precision",
                     "recall",
                     "acc",
                 ]):

        self.key = key
        self.trainer = trainer
        self.findings = findings
        self.metric_list = metric_list

        self.monitor = trainer.cfg["train"]["monitor"]["metric"]

        self.metric_dict = {
            f"{self.key}_{m}": {c: 0
                                for c in self.findings[self.key]}
            for m in self.metric_list
        }

        self.metric = Metric(self)
        self.outputs = []
        self.targets = []
        self.valid_step = 0

    def update(self, output, target):
        self.valid_step += 1

        output = output["cls"][self.key]
        output = output.cpu().detach().numpy()

        target = target["cls"][self.key]
        target = target.cpu().detach().numpy()

        self.outputs.append(output)
        self.targets.append(target)

    def evaluate(self):
        self.outputs = np.concatenate(self.outputs, axis=0)
        self.targets = np.concatenate(self.targets, axis=0)

        n = len(self.findings[self.key])
        for i in range(n):
            y_true = self.targets[:, i]
            y_prob = self.outputs[:, i]
            y_pred = self.outputs[:, i] > 0.5

            self.compute_metrics(y_true, y_pred, y_prob,
                                 self.findings[self.key][i])

        n = len(self.findings[self.key])
        for m in self.metric_dict:
            self.metric_dict[m]["total"] = 0.
            for c in self.findings[self.key]:
                self.metric_dict[m]["total"] += self.metric_dict[m][c] / n


        # self.outputs = self.outputs.reshape(-1)
        # self.targets = self.targets.reshape(-1)

        # y_true = self.targets
        # y_prob = self.outputs
        # y_pred = self.outputs > 0.5

        # self.compute_metrics(y_true, y_pred, y_prob, "Total")

        self.update_best_metric()
        self.write()
        self.reset()

    def compute_metrics(self, y_true, y_pred, y_prob, category):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.metric.compute_metrics(y_true, y_pred, y_prob, category)

    def write(self):
        writer = self.trainer.writer
        epoch = self.trainer.current_epoch
        # max_epoch = self.trainer.max_epoch

        colors = cycle(["green", "blue", "magenta", "cyan"])
        attrs = ["bold"]

        cprint(
            f"epoch {epoch} valid metrics",
            end=colored(":", "grey"),
            color=next(colors),
            attrs=attrs,
        )

        for m in self.metric_list:
            v = self.metric_dict[f"{self.key}_{m}"]["total"]
            cprint(
                f"{self.key}_{m} {v:.4f}",
                end=colored("||", "grey"),
                color=next(colors),
                attrs=attrs,
            )
            writer.add_scalars(
                f"valid/metrics/{self.key}_{m}",
                self.metric_dict[f"{self.key}_{m}"],
                global_step=epoch,
                walltime=time.time(),
            )

        trainer = self.trainer
        cfg = trainer.cfg["train"]["monitor"]

        best_metric = trainer.best_metric
        best_epoch = trainer.best_epoch
        metric = cfg['metric']

        if metric in self.metric_dict:
            cprint(
                f"best {metric} {best_metric:.4f} at epoch {best_epoch}",
                end=colored("||", "grey"),
                color="red",
                attrs=attrs,
            )

        cprint(
            f"task {trainer.args.task_name}",
            end=colored("||\n", "grey"),
            color=next(colors),
            attrs=attrs,
        )

    def update_best_metric(self):
        cfg = self.trainer.cfg["train"]["monitor"]
        if self.monitor in self.metric_dict:
            self.best_metric = self.metric_dict[cfg["metric"]]["total"]

            if cfg["mode"] == "min":
                if self.best_metric < self.trainer.best_metric:
                    self.trainer.best_metric = self.best_metric
                    self.trainer.best_epoch = self.trainer.current_epoch
                    self.trainer.save_model()
            else:
                if self.trainer.best_metric < self.best_metric:
                    self.trainer.best_metric = self.best_metric
                    self.trainer.best_epoch = self.trainer.current_epoch
                    self.trainer.save_model()

    def reset(self):
        self.__init__(
            trainer=self.trainer,
            key=self.key,
            findings=self.findings,
            metric_list=self.metric_list,
        )


class MonitorModeError(Exception):
    ...


class ValidLossLogger:
    def __init__(
        self,
        key,
        trainer,
    ):
        self.key = key
        self.trainer = trainer

        # self.loss_list = trainer.loss_list
        self.monitor = trainer.cfg["train"]["monitor"]["metric"]

        self.loss_dict = trainer.criterion.loss_dict.copy()
        self.valid_step = 0

    def update(self, output, target):
        loss_dict = self.trainer.criterion.loss_dict.copy()
        self.valid_step += 1

        if self.valid_step == 1:
            self.loss_dict = loss_dict
        else:
            for k in self.loss_dict:
                self.loss_dict[k] += loss_dict[k]

    def evaluate(self):
        for k in self.loss_dict:
            self.loss_dict[k] /= self.valid_step

        self.update_best_metric()
        self.write()
        self.reset()

    def write(self):
        writer = self.trainer.writer
        epoch = self.trainer.current_epoch

        colors = cycle(["green", "blue", "magenta", "cyan"])
        attrs = ["bold"]

        cprint(
            f"epoch {epoch} valid loss",
            end=colored(": ", "grey"),
            color=next(colors),
            attrs=attrs,
        )

        loss_dict = self.loss_dict

        for k in self.loss_dict:
            v = self.loss_dict[k]
            cprint(
                f"{k} {v:.4f}",
                end=colored("||", "grey"),
                color=next(colors),
                attrs=attrs,
            )

            writer.add_scalar(
                f"valid/loss/{k}",
                loss_dict[k],
                global_step=epoch,
                walltime=time.time(),
            )

        cfg = self.trainer.cfg["train"]["monitor"]

        trainer = self.trainer
        best_metric = trainer.best_metric
        best_epoch = trainer.best_epoch
        metric = cfg['metric']

        if metric in loss_dict:
            cprint(
                f"best {metric} {best_metric:.4f} at epoch {best_epoch}",
                end=colored("||", "grey"),
                color="red",
                attrs=attrs,
            )
        print("")

    def update_best_metric(self):
        cfg = self.trainer.cfg["train"]["monitor"]

        if self.monitor in self.loss_dict:
            self.best_metric = self.loss_dict[cfg["metric"]]

            if cfg["mode"] == "min":
                if self.best_metric < self.trainer.best_metric:
                    self.trainer.best_metric = self.best_metric
                    self.trainer.best_epoch = self.trainer.current_epoch
                    self.trainer.save_model()
            else:
                raise MonitorModeError(
                    "monitor mode should be min when monitor a loss.", )

    def reset(self):
        self.__init__(key=self.key, trainer=self.trainer)
