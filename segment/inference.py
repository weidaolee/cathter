import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from models import SMPModel, PostProcess
from datasets.segment import SegmentDataset
from losses.dice import DiceLoss

# set task prefix
# prefix = "seg_with_cls"
prefix = "seg_without_cls"
# checkpoint = "./results/seg_with_cls/checkpoints/model_0049.pth"
checkpoint = "./results/seg_without_cls/checkpoints/model_0049.pth"

# set device
device = "0"
device = [int(i) for i in list(device.split(","))]
device = device[0]

# get config
with open("./config.json", "r") as f:
    config = json.load(f)

model_config_dict = config["model"]

# load checkpoint
state_dict = torch.load(checkpoint)

# set save path and mkdir
result_base = f"../check/{prefix}"
os.makedirs(result_base, exist_ok=True)


# prepare model
def prepare_model():
    # prepare model

    print("Config model...")
    model = SMPModel(model_config_dict)()

    print(f"Push model to device {device}...")
    model.encoder = torch.nn.DataParallel(module=model.encoder,
                                          device_ids=[device])
    model.decoder = torch.nn.DataParallel(
        module=model.decoder,
        device_ids=[device],
    )
    model.classification_head = torch.nn.DataParallel(
        module=model.classification_head,
        device_ids=[device],
    )

    model = model.to(device)

    print("Laod model state...")
    model.load_state_dict(state_dict["model"])

    return model


if __name__ == "__main__":

    model = prepare_model()

    # prepare criterion
    print("Prepare criterion...")
    criterion = DiceLoss(smooth=config["loss"]["dice"]["smooth"])

    # prepare data pipeline
    print("Prepare data pipeline...")
    dataset = SegmentDataset("./data/valid_seg_tab.csv", is_train=False)

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        prefetch_factor=1,
    )

    postprocess = PostProcess([1, 1])

    model.eval()

    print("Start inference...")
    dice_list = []
    with torch.no_grad():
        for row, image, mask, finding in tqdm(dataloader, total=len(dataloader)):
            image = image.to(device)
            target = {
                "segment": mask.to(device, torch.float32),
                "classify": finding.to(device, torch.float32),
            }

            # inference
            output = model(image)
            output = {
                "segment": output[0],
                "classify": output[1],
            }
            # post-process
            output["segment"], target["segment"] = postprocess(
                output["segment"],
                target["segment"],
            )

            _ = criterion(output, target)

            dice_score = 1 - criterion.loss_dict["dice"].item()
            dice_list.append(dice_score)

            arr = np.zeros([512, 512 * 3])
            image = image.cpu().detach().numpy()
            image = np.squeeze(image)
            image = (image - image.min()) / (image.max() - image.min())
            arr[:, 512 * 0:512 * 1] = image[5, ...]

            output = output["segment"]
            output = torch.sigmoid(output)
            output = output.cpu().detach().numpy()
            output = np.squeeze(output)
            arr[:, 512 * 1:512 * 2] = output

            mask = mask.cpu().detach().numpy()
            mask = np.squeeze(mask)
            arr[:, 512 * 2:512 * 3] = mask

            uid = row["StudyInstanceUID"][0]

            save_path = os.path.join(result_base, uid)
            save_path = save_path + ".png"
            plt.imsave(save_path, arr, cmap="gray")
