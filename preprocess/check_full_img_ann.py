import os
import tqdm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from skimage.morphology import dilation, binary_dilation, disk
from skimage.transform import resize

tab = pd.read_csv("./data/train_preprocessed_tab.csv")
ann = pd.read_csv("./data/train_preprocessed_ann.csv")

pool = set(tab["StudyInstanceUID"].tolist())


def in_pool(uid):
    if uid in pool:
        return True
    else:
        return False


uids = ann["StudyInstanceUID"]
ann["Had Preprocessed"] = uids.apply(lambda x: in_pool(x))
ann = ann[ann["Had Preprocessed"] == True]

base_dir = "./data/train"

uids = ann["StudyInstanceUID"]
ann["Path"] = uids.apply(lambda u: os.path.join(base_dir, u))

os.makedirs("./vis", exist_ok=True)


class Plot:
    def __init__(self, size=512):
        self.size = size

    def __call__(self, row):
        row = row[1]
        self.res = np.zeros([self.size, self.size * 4])

        ori_img = self.load_ori_image(row)
        eql_img = self.load_eql_image(row)
        gth_img = self.make_mask(row)

        self.make_mark(row)

        ori_img = self.apply_mark(ori_img)
        gth_img = self.apply_mark(gth_img * 255)
        eql_0 = self.apply_mark(eql_img[..., 4])
        eql_1 = self.apply_mark(eql_img[..., 6])

        self.plot(ori_img, 0)
        self.plot(eql_0, 1)
        self.plot(gth_img, 2)
        self.plot(eql_1, 3)

        self.save_image(row)
        plt.close()

        return self.res

    def load_ori_image(self, row):
        p = row["Path"] + ".jpg"
        arr = plt.imread(p)

        ori_y = row["Original Y"]
        ori_x = row["Original X"]
        shift_y = row["Shift Y"]
        shift_x = row["Shift X"]

        new_arr = np.zeros([3827, 3567])

        new_arr[shift_y:shift_y + ori_y, shift_x:shift_x + ori_x] = arr

        new_arr = resize(new_arr, [self.size, self.size],
                         order=1,
                         preserve_range=True)

        return new_arr

    def load_eql_image(self, row):
        p = os.path.join(row["Path"], "image_9c.npz")
        arr = np.load(p)["image"]

        arr = resize(arr, [self.size, self.size], order=1, preserve_range=True)

        return arr

    def make_mask(self, row):
        arr = np.zeros([3827, 3567])
        seg = np.array(eval(row["Interpolated Points"]))

        for p in seg:
            arr[p[0], p[1]] = 1

        arr = dilation(arr, disk(5))

        arr = resize(arr, [self.size, self.size], order=0, preserve_range=True)

        return arr

    def make_mark(self, row):
        points = np.array(eval(row["data"]))

        points[:, 0] += row["Shift Y"]
        points[:, 1] += row["Shift X"]

        new_arr = np.zeros([3827, 3567])

        for p in points:
            new_arr[p[0], p[1]] = 1

        new_arr = binary_dilation(new_arr, disk(15))
        new_arr = resize(new_arr, [self.size, self.size],
                         order=0,
                         preserve_range=True)

        self.mark = new_arr

    def apply_mark(self, arr):
        arr[self.mark == 1] = 0

        return arr

    def plot(self, arr, i):
        res = self.res
        res[:, i * self.size:(i + 1) * self.size] = arr

        self.res = res

    def save_image(self, row):
        p = os.path.join("./vis", row["StudyInstanceUID"] + ".jpg")
        plt.imsave(p, self.res, cmap="gray")


for row in tqdm.tqdm(ann.iloc[:10].iterrows(), total=10):
    plot = Plot()
    res = plot(row)
