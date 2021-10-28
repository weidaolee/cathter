import time
import torch
import warnings
import datetime

import numpy as np
import sklearn.metrics as metrics


class TrainLogger:
    def __init__(self, trainer):
        self.trainer = trainer

    def update(self):
        step = self.trainer.global_step

        if step % 1 == 0:
            trainer = self.trainer
            writer = trainer.writer

            max_step = trainer.max_step
            epoch = trainer.current_epoch
            max_epoch = trainer.max_epoch

            lr = trainer.scheduler.get_lr()[0]

            loss_dict = self.trainer.criterion.loss_dict
            loss = self.trainer.criterion.loss_dict["Total"]

            current_time = datetime.datetime.now()
            spent_time = current_time - trainer.start_time
            spent_time = str(spent_time).split(".")[0]

            best_auc = trainer.best_auc
            best_epoch = trainer.best_epoch

            print(f"[Epoch {epoch:03d} | {max_epoch}]", end="  ")
            print(f"[Stpe {step:06d} | {max_step}]", end="  ")
            print(f"[lr {lr:.6f}]", end="  ")
            print(f"[Time {spent_time}]", end="  ")
            print(f"[Loss {loss:.6f}]", end="  ")
            print(f"[Best AUC: {best_auc:.4f} at epoch: {best_epoch}]")

            writer.add_scalars("running/loss", loss_dict, step, time.time())


class OverallValidLogger:
    def __init__(self, trainer):
        self.trainer = trainer
        self.outputs = []
        self.targets = []

        self.categories = [
            'ETT - Abnormal',
            'ETT - Borderline',
            'ETT - Normal',
            'NGT - Abnormal',
            'NGT - Borderline',
            'NGT - Incompletely Imaged',
            'NGT - Normal',
            'CVC - Abnormal',
            'CVC - Borderline',
            'CVC - Normal',
            'Swan Ganz Catheter Present',
        ]

        self.metric_dict = {
            m: {c: 0
                for c in self.categories + ["Total"]}
            for m in ["auc", "f1", "loss", "precision", "recall", "acc"]
        }

        self.loss_dict = {k: 0. for k in trainer.criterion.loss_dict.keys()}
        self.step = 0

    def update(self, output, target):
        self.step += 1
        loss_dict = self.trainer.criterion.loss_dict

        output = torch.sigmoid(output)
        output = output.cpu().detach().numpy()
        target = target.cpu().detach().numpy()

        self.outputs.append(output)
        self.targets.append(target)

        for k in self.loss_dict.keys():
            self.loss_dict[k] += loss_dict[k]

    def evaluate(self):
        self.outputs = np.concatenate(self.outputs, axis=0)
        self.targets = np.concatenate(self.targets, axis=0)

        for i in range(len(self.categories)):
            y_true = self.targets[:, i]
            y_prob = self.outputs[:, i]
            y_pred = self.outputs[:, i] > 0.5

            self.compute_metrics(y_true, y_pred, y_prob, self.categories[i])

        self.outputs = self.outputs.reshape(-1)
        self.targets = self.targets.reshape(-1)

        y_true = self.targets
        y_prob = self.outputs
        y_pred = self.outputs > 0.5

        self.compute_metrics(y_true, y_pred, y_prob, "Total")

        self.update_auc()
        self.write()

        self.reset()

    def write(self):
        writer = self.trainer.writer
        epoch = self.trainer.current_epoch
        max_epoch = self.trainer.max_epoch
        lr = self.trainer.scheduler.get_last_lr()

        writer.add_scalar("lr", lr, epoch)
        print(f"Eval [{epoch:03d} | {max_epoch}]", end=" ")

        print("[", end="")
        for m in ["auc", "f1", "loss", "precision", "recall", "acc"]:
            v = self.metric_dict[m]["Total"]
            print(f"{m}: {v:.4f}", end=" ")
            writer.add_scalars(f"valid/{m}",
                               self.metric_dict[m],
                               global_step=epoch,
                               walltime=time.time())
        print("]", end=" ")

        current_time = datetime.datetime.now()
        spent_time = current_time - self.trainer.start_time
        spent_time = str(spent_time).split(".")[0]

        best_auc = self.trainer.best_auc
        best_epoch = self.trainer.best_epoch
        print(f"[Time {spent_time}]", end=" ")
        print(f"[Best auc: {best_auc:.4f} at epoch: {best_epoch}]")

    def update_auc(self):
        self.auc = self.metric_dict["auc"]["Total"]
        if self.trainer.best_auc < self.auc:
            self.trainer.best_auc = self.auc
            self.trainer.best_epoch = self.trainer.current_epoch

            print("Save model...")
            self.trainer.save_model()

    def compute_metrics(self, y_true, y_pred, y_prob, category):
        warnings.filterwarnings("ignore")
        auc = metrics.roc_auc_score(y_true, y_prob)
        f1 = metrics.f1_score(y_true, y_pred)
        precision = metrics.precision_score(y_true, y_pred)
        recall = metrics.recall_score(y_true, y_pred)
        acc = metrics.accuracy_score(y_true, y_pred)

        c = category
        self.metric_dict["auc"][c] = auc
        self.metric_dict["f1"][c] = f1
        self.metric_dict["loss"][c] = self.loss_dict[c] / self.step
        self.metric_dict["precision"][c] = precision
        self.metric_dict["recall"][c] = recall
        self.metric_dict["acc"][c] = acc

    def reset(self):
        self.__init__(self.trainer)


class ExistLogger(OverallValidLogger):
    def __init__(self, trainer):
        super(ExistLogger, self).__init__(trainer)
        self.categories = [
            "ETT",
            "NGT",
            "CVC",
            "Swan Ganz Catheter Present",
        ]

        self.metric_dict = {
            m: {c: 0
                for c in self.categories + ["Total"]}
            for m in ["auc", "f1", "loss", "precision", "recall", "acc"]
        }

        self.loss_dict = {k: 0. for k in trainer.criterion.loss_dict.keys()}
        self.step = 0
