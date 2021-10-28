import os
from models.resnet200 import ResNet200D

import tqdm
import torch

import pandas as pd
import numpy as np

from collections import OrderedDict
from torch.utils.data import DataLoader
from datasets.inference import InferenceDataset


class Inferencer:
    def __init__(self, model, csv_path, ckpt_path, save_path, categories):
        self.model = model
        self.columns = categories
        self.dataset = InferenceDataset(csv_path)
        self.checkpoint = torch.load(ckpt_path)
        self.save_path = save_path

        self.outputs = []
        self.rows = []

        self.set_device()
        self.configuring()
        self.prepare_checkpoint()
        self.prepare_model()
        self.prepare_datapipeline()

    def set_device(self):
        # chose divice
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        self.workers = 48

    def prepare_checkpoint(self):
        model_dict = OrderedDict()
        for k, v in self.checkpoint["model"].items():
            model_dict[k.replace("module.", "")] = v

        self.model_dict = model_dict

    def prepare_model(self):
        self.model.load_state_dict(self.model_dict)
        self.model.eval()
        self.model = self.model.to(self.device)
        # self.model = torch.nn.DataParallel(self.model)

    def configuring(self):
        self.batch_size = 1

    def prepare_datapipeline(self):
        self.data_loader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            num_workers=self.workers,
        )

    def infer(self):
        print("Start infering...")
        with torch.no_grad():
            length = len(self.data_loader)
            for image, row in tqdm.tqdm(self.data_loader, total=length):
                image = image.to(self.device)
                output = self.model(image)

                self.update(output, row)

        self.merge()
        self.save_result()

    def update(self, output, row):
        output = torch.sigmoid(output)
        output = output.cpu().detach().numpy()

        row = pd.DataFrame(row)

        self.outputs.append(output)
        self.rows.append(row)

    def merge(self):
        self.outputs = np.concatenate(self.outputs, axis=0)
        self.outputs = pd.DataFrame(self.outputs, columns=self.columns)

        self.rows = pd.concat(self.rows, ignore_index=True)
        self.rows = self.rows[["Path", "StudyInstanceUID"]]

        self.result = pd.concat([self.rows, self.outputs], axis=1)

    def save_result(self):
        self.result.to_csv(self.save_path, index=False)


if __name__ == "__main__":

    os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 4, 5, 6, 7"
    model = ResNet200D(out_dim=11)
    csv_path = "./data/submission.csv"
    ckpt_path = "./results/overall/checkpoints/model_024.pth"
    save_path = "./results/overall/checkpoints/model_024.csv"
    categories = [
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

    inferencer = Inferencer(
        model=model,
        csv_path=csv_path,
        ckpt_path=ckpt_path,
        save_path=save_path,
        categories=categories,
    )
    inferencer.infer()
