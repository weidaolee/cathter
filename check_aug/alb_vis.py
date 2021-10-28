import os
import numpy as np
import pandas as pd

from random import shuffle

import cv2

import albumentations as alb

import matplotlib.pyplot as plt

df = pd.read_csv("./data/train_seg_tab.csv")

for i in range(df.shape[0]):
    row = df.iloc[i]

    findings = [
        'ETT - Abnormal', 'ETT - Borderline', 'ETT - Normal', 'NGT - Abnormal',
        'NGT - Borderline', 'NGT - Incompletely Imaged', 'NGT - Normal',
        'CVC - Abnormal', 'CVC - Borderline', 'CVC - Normal',
        'Swan Ganz Catheter Present'
    ]
    print(i, row[findings].sum())
    if row[findings].sum() == 5:
        break

image = np.load(os.path.join(row["Path"], "image_9c_512.npz"))["image"]
mask = np.load(os.path.join(row["Path"], "merged_label.npz"))["label"]

mask = mask.astype(np.int16)

size = 512

image_transforms = alb.Compose([
    alb.RandomBrightnessContrast(
        brightness_by_max=0.025,
        contrast_limit=0.025,
        always_apply=True,
    ),
    # alb.MotionBlur(
    #     blur_limit=3,
    #     always_apply=True,
    # ),
])

pair_transform = alb.Compose([
    alb.ShiftScaleRotate(
        shift_limit=0.075,
        scale_limit=0.1,
        rotate_limit=180,
        shift_limit_x=0.075,
        shift_limit_y=0.075,
        border_mode=cv2.BORDER_CONSTANT,
        always_apply=True,
        # mask_value=1,
    ),
    alb.ElasticTransform(
        alpha=5,
        alpha_affine=12,
        border_mode=cv2.BORDER_CONSTANT,
        always_apply=True,
        # mask_value=1,
    ),
    alb.GridDistortion(
        num_steps=10,
        distort_limit=0.05,
        border_mode=cv2.BORDER_CONSTANT,
        always_apply=True,
        # mask_value=1,
    ),
    alb.OpticalDistortion(
        distort_limit=0.05,
        shift_limit=0.05,
        border_mode=cv2.BORDER_CONSTANT,
        always_apply=True,
        # mask_value=1,
    ),
    alb.IAAPerspective(
        scale=(0.00, 0.05),
        always_apply=True,

    ),
])

aug_image = image_transforms(image=image)
aug_image, aug_mask= pair_transform(image=image, mask=mask)

os.makedirs("./aug_image", exist_ok=True)
for i in range(9):
    p = os.path.join("./aug_image", f"{i}.jpg")
    plt.imsave(p, aug_image["image"][..., i], cmap="gray")

p = os.path.join("./aug_image", "mask.jpg")
plt.imsave(p, aug_image["mask"], cmap="gray")
