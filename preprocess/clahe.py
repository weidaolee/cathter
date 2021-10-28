import os
import itertools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from skimage import exposure
from skimage.transform import resize


class CLAHE:
    def __init__(self, size=512):
        self.ori_img = None
        self.eql_img = []
        self.kernel_size = [0.02, 0.05, 0.10]
        self.clip_limit = [0.05, 0.1, 0.5]
        self.params = list(itertools.product(self.kernel_size,
                                             self.clip_limit))
        self.size = size

    def __call__(self, row):
        # try:
        #     row = row[1]
        #     self.read(row["Path"])

        #     params = self.params
        #     for p in params:
        #         img = self.perform_CLAHE(p)
        #         self.eql_img.append(img)

        #     self.eql_img = np.stack(self.eql_img, -1)

        #     self.save_image(row)

        #     return None

        # except:
        #     return row

        row = row[1]
        self.read(row["Path"])

        params = self.params
        for p in params:
            img = self.perform_CLAHE(p)
            self.eql_img.append(img)

        self.eql_img = np.stack(self.eql_img, -1)

        self.save_image(row)

    def read(self, path):
        img = plt.imread(path + ".jpg").astype(np.float)

        self.max = img.max()
        self.min = img.min()

        # normalization for CLAME
        img = ((img - self.min) / (self.max - self.min))

        self.ori_img = img
        return img

    def perform_CLAHE(self, params):
        ksize, clip_limit = params[0], params[1]
        kernel_size = np.array(self.ori_img.shape) * ksize // 1

        img = exposure.equalize_adapthist(image=self.ori_img,
                                          kernel_size=kernel_size,
                                          clip_limit=clip_limit)

        # denormalization for dcm formate
        img *= 255

        # to int16
        img = img.astype(np.int16)

        return img

    def save_image(self, row):

        path = os.path.join(row["Path"], "image_9c.npz")
        self.eql_img = self.to_uint8(self.eql_img)
        np.savez_compressed(path, image=self.eql_img)

        path = os.path.join(row["Path"], f"image_9c_{self.size}.npz")
        self.new_img = self.resize(self.eql_img)
        self.new_img = self.to_uint8(self.new_img)
        np.savez_compressed(path, image=self.new_img)

    def to_uint8(self, image):

        for i in range(9):
            img = image[..., i]
            img = (img - img.min()) / (img.max() - img.min())
            img *= 255
            image[..., i] = img
        image = image.astype(np.uint8)
        return image

    def resize(self, image):
        image = resize(image, [self.size, self.size],
                       order=3,
                       preserve_range=True)

        return image

p = "./data/train/1.2.826.0.1.3680043.8.498.12662459268358115142748448763853146057"
if __name__ == "__main__":
    
    tab = pd.read_csv("./data/train_preprocessed_tab.csv")
    tab = tab[tab["Path"] == p]
    print(tab)

    clahe = CLAHE()
    for row in tab.iterrows():
        clahe(row)
        break
