import os
import torch
import warnings

import numpy as np
import pandas as pd

import torchvision.transforms as transforms


class InferenceDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image = self.load_image(row)
        image = self.trasform(image)

        return image, row.to_dict()

    def trasform(self, image):
        warnings.filterwarnings("ignore")
        tensor = transforms.ToTensor()(image)

        tensor = self.standarize(tensor)
        return tensor

    def __len__(self):
        return self.df.shape[0]

    def load_image(self, row):
        path = os.path.join(row["Path"], "image_9c_512.npz")
        image = np.load(path, allow_pickle=True)["image"].astype(np.float32)
        return image

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
    dataset = InferenceDataset("./data/test_preprocessed_tab.csv")

    dataloader = DataLoader(dataset,
                            batch_size=128,
                            shuffle=False,
                            num_workers=1,
                            prefetch_factor=1)
    for i, batch in enumerate(dataloader):
        break
