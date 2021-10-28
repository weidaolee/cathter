import os
import tqdm
import json
import argparse

import torch
import numpy as np
import pandas as pd

from functools import partial
from multiprocessing.pool import ThreadPool
from torch.utils.data import DataLoader

from models.smp import SegmentModel
from datasets.inference import InferenceDataset
from utils.postprocess import DilateSigmoidPostProcess
from utils.postprocess import DilateSoftmaxPostProcess


def predict(Model, Dataset):
    args = parse_args()
    seg_prefix = args.seg_prefix
    cls_prefix = args.cls_prefix
    with open(args.config_path) as f:
        cfg = json.load(f)
    cfg["model"]["load_ckpt"] = True
    assert cfg["model"]["load_ckpt"]

    model = get_model(cfg, Model)
    dataset = get_dataset(args, cfg, Dataset)

    if "DilateSigmoidPostProcess" in cfg["postprocess"]:
        postprocess = DilateSigmoidPostProcess([1, 1])

    elif "DilateSoftmaxPostProcess" in cfg["postprocess"]:
        postprocess = DilateSoftmaxPostProcess([1, 1])

    Predictor(
        args=args,
        cfg=cfg,
        model=model,
        dataset=dataset,
        postprocess=postprocess,
        seg_prefix=seg_prefix,
        cls_prefix=cls_prefix,
    )


def get_model(cfg, Model):
    print(cfg["model"])
    model = Model(cfg["model"])
    return model


def get_dataset(args, cfg, Dataset):
    print("Prepare dataset...")
    print(cfg["dataset"])
    cfg["dataset"]["mask_file_postfix"] = ""

    valid_path = args.data_path
    valid_set = Dataset(
        valid_path,
        **cfg["dataset"],
        is_train=False,
    )
    return valid_set


class Predictor:
    def __init__(
        self,
        args,
        cfg,
        model,
        dataset,
        postprocess,
        seg_prefix,
        cls_prefix,
    ):
        print("Setting Arguments...", args)
        self.args = args
        self.cfg = cfg
        self.seg_prefix = seg_prefix
        self.cls_prefix = cls_prefix
        self.cls_output = []
        self.cls_target = []
        self.path_list = []

        self.data_path = self.args.data_path

        self.batch_size = 64

        self.set_device()
        self.load_checkpoint()
        self.prepare_model(model, postprocess)
        self.prepare_datapipeline(dataset)
        self.predict()

    def set_device(self):
        self.device = "cuda"

    def load_checkpoint(self):
        if self.args.checkpoint is not None:
            print("Load checkpoint...", self.args.checkpoint)
            self.state = torch.load(self.args.checkpoint)
        else:
            print("No checkpoint will be loaded.")

    def prepare_model(self, model, postprocess):
        print("Prepare model...")
        print("Model configuring...")
        cfg = self.cfg["model"]

        self.model = model

        self.model.encoder = torch.nn.DataParallel(
            self.model.encoder,
            # device_ids=self.device_ids,
        )

        if hasattr(model, "decoder"):
            self.model.decoder = torch.nn.DataParallel(
                self.model.decoder,
                # device_ids=self.device_ids,
            )

        self.model.classification_head = torch.nn.DataParallel(
            self.model.classification_head,
            # device_ids=self.device_ids,
        )

        self.model = self.model.to(self.device, dtype=torch.float32)

        if cfg["load_ckpt"]:
            print("Load model state...")
            self.model.load_state_dict(self.state["model"], strict=False)
        else:
            print("No model state dict will be loaded.")

        self.postprocess = postprocess

    def prepare_datapipeline(self, dataset):
        self.dataset = dataset

        # prepare dataloader
        print("Prepare dataloader...")
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=32,
            drop_last=False,
            prefetch_factor=1,
        )

    def predict(self):

        print("Start predicting...")
        with torch.no_grad():
            self.model.eval()
            num_batchs = len(self.dataloader)
            for row, inputs, target in tqdm.tqdm(self.dataloader,
                                                 total=num_batchs):

                for k in inputs:
                    inputs[k] = inputs[k].to("cuda")

                # inference
                output = self.model(inputs)

                # post-process
                output, target = self.postprocess(output, target)

                self.save_img(row, output, target)
            self.save_cls()

    def save_img(self, row, output, target):

        output["seg"] = output["seg"].cpu().detach().numpy()

        for k in output["cls"]:
            output["cls"][k] = output["cls"][k].cpu().detach().numpy()
            self.cls_output.append(output["cls"][k])
            self.cls_target.append(target["cls"][k])
            self.path_list.extend(row["Path"])

        _save = partial(self._save_img, row=row, output=output)

        with ThreadPool(min(self.batch_size, 64)) as pool:
            gen = pool.imap(_save, range(output["seg"].shape[0]))
            for _ in gen:
                ...

    def _save_img(self, i, row, output):
        pred = output["seg"][i, ...]
        pred = np.squeeze(pred)
        pred = pred.astype(np.float32)

        # save_path = os.path.join(row["Path"][i], "pred_ett.npz")
        # np.savez_compressed(save_path, arr=pred[0, ...])

        # save_path = os.path.join(row["Path"][i], "pred_ngt.npz")
        # np.savez_compressed(save_path, arr=pred[1, ...])

        save_path = os.path.join(row["Path"][i], "pred_cvc.npz")
        np.savez_compressed(save_path, arr=pred[2, ...])

    def save_cls(self):
        cfg = self.cfg["dataset"]["findings"]
        task_name = self.args.task_name
        for k in cfg:

            output = np.concatenate(self.cls_output, axis=0)
            output = pd.DataFrame(output, columns=cfg[k])

            target = np.concatenate(self.cls_target, axis=0)
            target = pd.DataFrame(target, columns=cfg[k])

            output["Path"] = self.path_list
            target["Path"] = self.path_list

            output.to_csv(
                f"./results/{task_name}/{self.cls_prefix}_output.csv",
                index=False)
            target.to_csv(
                f"./results/{task_name}/{self.cls_prefix}_target.csv",
                index=False)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seg_prefix",
        type=str,
    )
    parser.add_argument(
        "--cls_prefix",
        type=str,
    )
    parser.add_argument(
        "--task_name",
        type=str,
    )
    parser.add_argument(
        '--gpus',
        type=str,
        help='specify which GPUs to use.',
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=32,
        help='number of workers',
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='pytorch checkpoint file path',
    )
    parser.add_argument(
        "--config_path",
        type=str,
    )
    parser.add_argument(
        "--data_path",
        type=str,
    )

    return parser.parse_args()


if __name__ == "__main__":
    predict(SegmentModel, InferenceDataset)
