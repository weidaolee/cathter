import os

import numpy as np
import pandas as pd

import torchvision.transforms as transforms

import matplotlib.pyplot as plt

tab = pd.read_csv("./data/train_preprocessed_tab.csv")

row = tab.iloc[0]

path = os.path.join(row["Path"], "image_9c_512.npz")

image = np.load(path)["image"]

transform = transforms.Compose([
    transforms.RandomPerspective(distortion_scale=0.1),
    transforms.RandomRotation([-12, 12]),
    transforms.RandomResizedCrop(size=[512, 512],
                                 scale=[0.95, 1.00],
                                 ratio=[0.95, 1.05],
                                 interpolation=3),
])

image = transforms.ToTensor()(image)
image = transform(image)

image = image.numpy()

image = np.transpose(image, [1, 2, 0])
base_dir = "check_img"
os.makedirs(base_dir, exist_ok=True)
for i in range(9):
    p = os.path.join(base_dir, f"{i}.jpg")
    plt.imsave(p, image[..., i], cmap="gray")
