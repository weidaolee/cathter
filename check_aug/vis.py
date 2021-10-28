import os
import numpy as np
import pandas as pd

from random import shuffle

import imgaug.augmenters as iaa

import matplotlib.pyplot as plt

df = pd.read_csv("./data/train_preprocessed_tab.csv")
row = df.iloc[254]
path = os.path.join(row["Path"], "image_9c_512.npz")
image = np.load(path)["image"].astype(np.float16)

augmenters = [
    # iaa.AdditiveGaussianNoise(scale=(0, 0.02 * 255), per_channel=True),
    iaa.Multiply((0.98, 1.02), per_channel=True),
    iaa.MultiplyElementwise((0.98, 1.02), per_channel=True),
    iaa.GammaContrast((0.8, 1.2), per_channel=True,),
]



# seq = iaa.Sometimes(p=0.5, then_list=augmenters)
seq = iaa.Sequential(augmenters)

image_aug = seq(images=image)
image_aug = np.clip(image_aug, 0, 255)
# image_aug = (image_aug - image_aug.min()) / (image_aug.max() - image_aug.min())
# image_aug *= 255
# image_aug = image_aug.astype(np.uint8)


os.makedirs("./aug_image", exist_ok=True)
for i in range(9):
    p = os.path.join("./aug_image", f"{i}.jpg")
    plt.imsave(p, image_aug[..., i], cmap="gray")
