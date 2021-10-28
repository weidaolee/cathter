import os
import tqdm

import multiprocessing

import numpy as np
import pandas as pd

from skimage.morphology import binary_dilation, disk
from skimage.transform import resize


class ImageProcessor:
    def __init__(self, size):
        self.arr = None

    def __call__(self, row):
        try:
            row = row[1]
            self.load_eql_image(row)
            self.resize()
            self.save_image(row)            

            return None

        except:
            print(row)
            return row

        # row = row[1]
        # self.load_eql_image(row)
        # # self.resize()
        # self.save_image(row)

    def load_eql_image(self, row):
        p = os.path.join(row["Path"], "image_9c.npz")
        img = np.load(p)["image"]
        self.ori_img = img

    def resize(self):
        img = resize(self.ori_img, [self.size, self.size],
                     order=3,
                     preserve_range=True)

        self.new_img = img

    def to_uint8(self, image):

        for i in range(9):
            img = image[..., i]
            img = (img - img.min()) / (img.max() - img.min())
            img *= 255
            image[..., i] = img
        image = image.astype(np.uint8)
        return image

    def save_image(self, row):
        # self.ori_img = self.to_uint8(self.ori_img)
        # p = os.path.join(row["Path"], "image_9c.npz")
        # np.savez_compressed(p, image=self.ori_img)

        p = os.path.join(row["Path"], f"image_9c_{self.size}.npz")
        self.new_img = self.to_uint8(self.new_img)
        np.savez_compressed(p, image=self.new_img)


def process_image():
    failed_list = []
    with multiprocessing.Pool(32) as pool:
        gen = pool.imap(ImageProcessor(size=size), tab.iterrows())
        for row in tqdm.tqdm(gen, total=tab.shape[0]):
            if row is not None:
                failed_list.append(row)

    if len(failed_list) != 0:
        df = pd.DataFrame(data=failed_list)
        df.to_csv("./failed_to_input_img.csv", index=True)


class LabelProcessor:
    def __init__(self, size):
        self.size = size
        self.arr = None

    def __call__(self, row):
        try:
            row = row[1]
            self.get_catheter_type(row)
            self.prepare_mask(row)

            self.save_label(row)

            return None

        except:
            return row

        # row = row[1]
        # self.get_catheter_type(row)
        # self.prepare_mask(row)
        # self.save_label(row)

    def get_catheter_type(self, row):
        self.type = None
        label = row["label"]
        for t in ["CVC", "NGT", "ETT", "Swan Ganz Catheter Present"]:
            if t in label:
                self.type = t.replace("Swan Ganz Catheter Present", "SGC")
                break

    def prepare_mask(self, row):
        arr = np.zeros([row["Original Y"], row["Original X"]]).astype(np.int8)
        seg = np.array(eval(row["Interpolated Points"]))

        for p in seg:
            try:
                arr[p[0], p[1]] = 1
            except:
                pass

        arr = binary_dilation(arr, disk(8))

        arr = resize(arr, [self.size, self.size], order=0, preserve_range=True)

        arr = arr.astype(bool)

        self.arr = arr

    def save_label(self, row):
        p = os.path.join(row["Path"], f"label_{self.type}.npz")
        s = 0
        while os.path.exists(p):
            p = os.path.join(row["Path"], f"label_{self.type}_{s}.npz")
            s += 1
        np.savez(p, label=self.arr)

    # def read_label(self, row):
    #     p = os.path.join(row["Path"], f"label_{self.type}.npz")
    #     self.arr = np.load(p)["label"].astype(bool).squeeze()
    #     np.savez(p, label=self.arr)


def process_label():
    failed_row = []
    with multiprocessing.Pool(54) as pool:
        gen = pool.imap(LabelProcessor(size=size), ann.iterrows())
        for row in tqdm.tqdm(gen, total=ann.shape[0]):
            if row is not None:
                print(row)
                failed_row.append(row)

    df = pd.DataFrame(data=failed_row)
    if df.shape[0] != 0:
        df.to_csv("./failed_label_path.csv", index=True)


size = 512
tab = pd.read_csv("./data/train_preprocessed_tab.csv")
ann = pd.read_csv("./data/train_preprocessed_ann.csv")
# ann = pd.read_csv("./failed_label_path.csv")

if __name__ == "__main__":
    # process_image()
    process_label()
