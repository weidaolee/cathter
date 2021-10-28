import numpy as np
import pandas as pd
import warnings

from torchvision import transforms

from datasets.base import BaseDataset


class InferenceDataset(BaseDataset):
    def __init__(
        self,
        csv_path,
        findings,
        image_file_postfix="image_9c_512",
        mask_file_postfix="",
        catheters=None,
        is_train=False,
        precache=1,
    ):
        super(InferenceDataset, self).__init__(
            csv_path=csv_path,
            findings=findings,
            image_file_postfix=image_file_postfix,
            mask_file_postfix=mask_file_postfix,
            catheters=catheters,
            is_train=is_train,
            precache=precache,
        )

        self.df = self.df.drop_duplicates()

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image, mask = super().__getitem__(idx)
        label = self.load_label(row)

        image = self.transform(image)
        inputs = {"image": image}
        target = {"cls": label}
        return row.to_dict(), inputs, target

    def transform(self, image):
        warnings.filterwarnings("ignore")

        image = image.astype(np.float32)
        image = transforms.ToTensor()(image)
        image = self.standarize(image)

        return image
