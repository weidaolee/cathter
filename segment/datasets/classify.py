import os
import torch
import warnings
import numpy as np

import torchvision

from datasets.base import BaseDataset


class ClassifyDataset(BaseDataset):
    def __init__(
        self,
        csv_path,
        findings,
        image_file_postfix="image_9c_512",
        mask_file_postfix="seg_pred",
        catheters="total",
        precache=0.5,
        is_train=False,
    ):
        super(ClassifyDataset, self).__init__(
            csv_path=csv_path,
            findings=findings,
            image_file_postfix=image_file_postfix,
            mask_file_postfix=mask_file_postfix,
            catheters=catheters,
            is_train=is_train,
            precache=precache,
        )


class SimpleClassifyDataset(ClassifyDataset):
    def __init__(
        self,
        csv_path,
        findings,
        image_file_postfix="baseline",
        mask_file_postfix="",
        catheters="total",
        precache=0.5,
        is_train=False,
    ):
        super(SimpleClassifyDataset, self).__init__(
            csv_path=csv_path,
            findings=findings,
            image_file_postfix=image_file_postfix,
            mask_file_postfix=mask_file_postfix,
            catheters=catheters,
            is_train=is_train,
            precache=precache,
        )


    def transform(self, image):
        warnings.filterwarnings("ignore")

        if self.is_train:
            aug = self.augmentations(image=image)
            image = aug["image"]

        image = image.astype(np.float32)
        image = torchvision.transforms.ToTensor()(image)
        image = self.standarize(image)

        return image

    def load_mask(self, row):
        return None

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image, mask = super().__getitem__(idx)

        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=-1)

        label = self.load_label(row)

        image = self.transform(image=image)
        inputs = {"image": image}

        target = {"cls": label}

        return row.to_dict(), inputs, target


class SegmentBasedClassifyDataset(ClassifyDataset):
    def __init__(
        self,
        csv_path,
        findings,
        image_file_postfix="image_9c_512",
        mask_file_postfix="seg_pred",
        catheters="total",
        precache=1.0,
        is_train=False,
    ):
        super(SegmentBasedClassifyDataset, self).__init__(
            csv_path=csv_path,
            findings=findings,
            image_file_postfix=image_file_postfix,
            mask_file_postfix=mask_file_postfix,
            catheters=catheters,
            precache=precache,
            is_train=is_train,
        )

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image, mask = super().__getitem__(idx)

        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=-1)

        image, mask = self.transform(image=image, mask=mask)

        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=-1)

        inputs = {"image": image, "pred": mask}

        label = self.load_label(row)
        target = {"cls": label}

        return row.to_dict(), inputs, target


if __name__ == "__main__":
    import json
    import matplotlib.pyplot as plt
    from torch.utils.data import DataLoader

    with open("./config.json") as f:
        cfg = json.load(f)

    cfg = cfg["dataset"]

    dataset = ClassifyDataset("./data/train_tab.csv", is_train=True, **cfg)

    dataloader = DataLoader(dataset,
                            batch_size=8,
                            shuffle=False,
                            num_workers=32,
                            prefetch_factor=1)

    for i, (_, image, target) in enumerate(dataloader):
        print(i)
        break
