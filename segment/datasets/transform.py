import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import cv2

import albumentations as alb

image_transforms = alb.Compose([
    alb.RandomBrightnessContrast(brightness_by_max=0.025,
                                 contrast_limit=0.025,
                                 always_apply=True,
                                 p=0),
])

pair_transforms = alb.Compose([
    alb.RandomBrightnessContrast(
        brightness_limit=0.2,
        contrast_limit=0.2,
        brightness_by_max=True,
        # p=self.p,
    ),
    alb.ShiftScaleRotate(
        shift_limit=0.075,
        scale_limit=0.1,
        rotate_limit=45,
        shift_limit_x=0.075,
        shift_limit_y=0.075,
        border_mode=cv2.BORDER_CONSTANT,
        # p=self.p,
    ),
    alb.ElasticTransform(
        alpha=5,
        alpha_affine=12,
        border_mode=cv2.BORDER_CONSTANT,
        # p=self.p,
    ),
    alb.GridDistortion(
        num_steps=10,
        distort_limit=0.05,
        border_mode=cv2.BORDER_CONSTANT,
        # p=self.p,
    ),
    alb.IAAPerspective(scale=(0.00, 0.05),
                       # p=self.p,
                       ),
    alb.ChannelShuffle(p=1),
    alb.HorizontalFlip(p=0.5),
])

path = "../data/train/1.2.826.0.1.3680043.8.498.10000428974990117276582711948006105617/"

# for i in range(9):
#     plt.imsave(f"./aug_image/{i}.png", image[..., i], cmap="gray")

for i in range(10):
    image = np.load(os.path.join(path, "image_9c_512.npz"))["image"]
    mask = np.load(os.path.join(path,
                                "merged_label.npz"))["label"].astype(np.int16)
    aug_pair = pair_transforms(image=image, mask=mask)
    image, mask = aug_pair["image"], aug_pair["mask"]

    plt.imsave(f"./aug_image/{i}.png", image[..., 5], cmap="gray")
plt.imsave("./aug_image/mask.png", mask, cmap="gray")
