import os
import torch
import warnings

import numpy as np
import pandas as pd

import torchvision.transforms as transforms

from random import shuffle


class OverallDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path, is_train=False):
        self.df = pd.read_csv(csv_path)

        self.is_train = is_train

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

        self.transformations = torch.nn.ModuleList([
            transforms.RandomPerspective(distortion_scale=0.1),
            transforms.RandomRotation([-12, 12]),
            transforms.RandomResizedCrop(
                size=[512, 512],
                scale=[0.95, 1.00],
                ratio=[0.95, 1.05],
                interpolation=3,
            ),
        ])

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image = self.load_image(row)
        image = self.transform(image)

        label = self.load_label(row)

        return image, label

    def transform(self, image):
        warnings.filterwarnings("ignore")
        tensor = transforms.ToTensor()(image)

        if self.is_train:
            shuffle(self.transformations)
            image = transforms.RandomApply(self.transformations)(tensor)

        tensor = self.standarize(tensor)
        return tensor

    def __len__(self):
        return self.df.shape[0]

    def load_image(self, row):
        path = os.path.join(row["Path"], "image_9c_512.npz")
        image = np.load(path, allow_pickle=True)["image"].astype(np.float32)
        return image

    def load_label(self, row):
        label = row[self.categories].to_numpy().astype(np.float32)
        return label

    def standarize(self, tensor):
        mean = torch.mean(tensor, [1, 2], keepdims=True)
        std = torch.std(tensor, [1, 2], keepdims=True)

        tensor = (tensor - mean) / std

        return tensor

    def to_chennal_first(self, image):
        image = np.transpose(image, [2, 0, 1])
        return image


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    # dataset = OverallDataset("./data/train_preprocessed_tab.csv",
    #                          is_train=False)

    dataset = OverallDataset("./data/train_tab.csv", is_train=True)

    targets = []
    dataloader = DataLoader(dataset,
                            batch_size=4,
                            shuffle=False,
                            num_workers=4,
                            prefetch_factor=1)

    for i, batch in enumerate(dataloader):
        break
        targets.append(batch[1].numpy())

    targets = np.concatenate(targets, axis=0)
