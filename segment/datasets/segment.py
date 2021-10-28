import numpy as np
import warnings

import torchvision

from datasets.base import BaseDataset


class SegmentDataset(BaseDataset):
    def __init__(
        self,
        csv_path,
        findings,
        image_file_postfix="image_9c_512",
        mask_file_postfix="merged_label",
        catheters="total",
        is_train=False,
        precache=0.5,
    ):
        super(SegmentDataset, self).__init__(
            csv_path=csv_path,
            findings=findings,
            image_file_postfix=image_file_postfix,
            mask_file_postfix=mask_file_postfix,
            catheters=catheters,
            is_train=is_train,
            precache=precache,
        )

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image, mask = super().__getitem__(idx)
        label = self.load_label(row)

        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=-1)

        image, mask = self.transform(image=image, mask=mask)

        inputs = {"image": image}
        target = {"seg": mask, "cls": label}

        return row.to_dict(), inputs, target




if __name__ == "__main__":
    import json
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    with open("./config.json") as f:
        cfg = json.load(f)

    cfg = cfg["dataset"]

    dataset = SegmentDataset("./data/train_seg_tab.csv", is_train=True, **cfg)

    dataloader = DataLoader(dataset,
                            batch_size=1,
                            shuffle=False,
                            num_workers=1,
                            prefetch_factor=1)

    for i, (_, image, target) in enumerate(dataloader):
        print(i)
        break

    # target = mask[10, ...].numpy()
    # target = target.squeeze()
    # plt.imsave("target.jpg", target * 255, cmap="gray")

    # targets = np.concatenate(targets, axis=0)
