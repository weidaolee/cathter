import os
import tqdm
import torch
import torchvision
import warnings

import numpy as np
import pandas as pd
# import cv2
import albumentations as alb
# import torchvision.transforms as transforms
from multiprocessing import cpu_count
from multiprocessing.pool import ThreadPool
from utils.timer import TickTock


class BaseDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        csv_path,
        findings,
        image_file_postfix="image_9c_512",
        mask_file_postfix="merged_label",
        is_train=False,
        classification_head=True,
        catheters="total",
        precache=1.0,
    ):
        self.df = self.read_csv(csv_path, catheters)
        self.is_train = is_train
        self.image_file_postfix = image_file_postfix
        self.mask_file_postfix = mask_file_postfix

        self.findings = findings

        self.image_pool = {}
        self.mask_pool = {}

        self.precache_data(ratio=precache)

        print(f"image pool: {len(self.image_pool)}")
        print(f"mask pool: {len(self.mask_pool)}")

        if is_train:
            self.augmentations = alb.Compose([])

    def read_csv(self, csv_path, catheters):
        usecols = [
            "StudyInstanceUID",
            "PatientID",
            "Path",
        ]

        if catheters is None:
            df = pd.read_csv(csv_path, usecols=usecols)

        else:
            catheters = catheters.lower()

        if catheters in ["ett", "total"]:
            usecols += [
                "ETT",
                "ETT - Abnormal",
                "ETT - Borderline",
                "ETT - Normal",
            ]
            if catheters != "total":
                df = pd.read_csv(csv_path, usecols=usecols)
                df = df[df[catheters.upper()] == 1]

        if catheters in ["ngt", "total"]:
            usecols += [
                "NGT",
                "NGT - Abnormal",
                "NGT - Borderline",
                "NGT - Incompletely Imaged",
                "NGT - Normal",
            ]

            if catheters != "total":
                df = pd.read_csv(csv_path, usecols=usecols)
                df = df[df[catheters.upper()] >= 1]

        if catheters in ["cvc", "total"]:
            usecols += [
                "CVC",
                "CVC - Abnormal",
                "CVC - Borderline",
                "CVC - Normal",
            ]
            if catheters != "total":
                df = pd.read_csv(csv_path, usecols=usecols)
                df = df[df[catheters.upper()] == 1]

        if catheters == "total":
            usecols += ["Swan Ganz Catheter Present"]
            df = pd.read_csv(csv_path, usecols=usecols)

        return df

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        mask = None
        assert isinstance(row["Path"], str)

        if row["Path"] not in self.image_pool:
            image = self.load_image(row)
            if self.mask_file_postfix != "":
                mask = self.load_mask(row)

        else:
            image = self.image_pool[row["Path"]]
            if self.mask_file_postfix != "":
                mask = self.mask_pool[row["Path"]]

        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=-1)

        assert len(image.shape) < 4

        return image, mask

    @TickTock()
    def precache_data(self, ratio=0.5):
        df = self.df["Path"].value_counts(ascending=False).index
        df = pd.DataFrame(df, columns=["Path"])
        num = int(df.shape[0] * ratio)

        rows = [df.iloc[i] for i in range(num)]

        workers = cpu_count() // 2
        assert workers > 0
        with ThreadPool(workers) as pool:
            gen = pool.imap(self.__precache_image, rows)
            for path, image in tqdm.tqdm(gen, total=len(rows)):
                assert isinstance(path, str)
                self.image_pool[path] = image

        if self.mask_file_postfix != "":
            with ThreadPool(workers) as pool:
                gen = pool.imap(self.__precache_mask, rows)
                for path, mask in tqdm.tqdm(gen, total=len(rows)):
                    assert isinstance(path, str)
                    self.mask_pool[path] = mask

    def __len__(self):
        return self.df.shape[0]

    def load_image(self, row):
        path = os.path.join(row["Path"], f"{self.image_file_postfix}.npz")
        image = np.load(path, allow_pickle=True)["arr"].astype(np.uint8)
        return image

    def load_label(self, row):
        label = {}
        for k in self.findings:
            label[k] = row[self.findings[k]].to_numpy()
            label[k] = (label[k] >= 1).astype(np.float32)
        return label

    def load_mask(self, row):
        path = os.path.join(row["Path"], f"{self.mask_file_postfix}.npz")
        mask = np.load(path, allow_pickle=True)["arr"].astype(np.float32)
        return mask

    def transform(self, image, mask):
        warnings.filterwarnings("ignore")

        if self.is_train:
            aug = self.augmentations(image=image, mask=mask)
            image, mask = aug["image"], aug["mask"]

        image = image.astype(np.float32)
        image = torchvision.transforms.ToTensor()(image)
        image = self.standarize(image)

        if len(mask.shape) < 3:
            mask = np.expand_dims(mask, -1)

        mask = mask.astype(np.float32)
        mask = torchvision.transforms.ToTensor()(mask)

        return image, mask

    def standarize(self, tensor):
        mean = torch.mean(tensor, [1, 2], keepdims=True)
        std = torch.std(tensor, [1, 2], keepdims=True)

        tensor = (tensor - mean) / std

        return tensor

    def to_chennal_first(self, arr):
        image = np.transpose(arr, [2, 0, 1])
        return image

    def __precache_image(self, row):
        image = self.load_image(row)
        return row["Path"], image

    def __precache_mask(self, row):
        try:
            mask = self.load_mask(row)
        except FileNotFoundError:
            mask = np.zeros([512, 512])
        return row["Path"], mask
