import tqdm

import multiprocessing
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from clahe import CLAHE


def _get_ori_shape(p):
    img = plt.imread(p + ".jpg")
    return img.shape


def get_ori_shape(path_list, worker=32):
    shape_list = []
    with multiprocessing.Pool(worker) as pool:
        gen = pool.imap(_get_ori_shape, path_list)
        for shape in tqdm.tqdm(gen, total=len(path_list)):
            shape_list.append(shape)

    return shape_list


def expand_table_info(table):
    shape_list = get_ori_shape(tab["Path"].to_list())
    shape_list = np.array(shape_list)
    tab["Original Y"] = shape_list[:, 0]
    tab["Original X"] = shape_list[:, 1]

    tab.to_csv("./data/test_preprocessed_tab.csv", index=False)


def process_image(worker):
    with multiprocessing.Pool(worker) as pool:
        failed_list = []
        gen = pool.imap(CLAHE(), tab.iterrows())

        for row in tqdm.tqdm(gen, total=tab.shape[0]):
            if row is not None:
                failed_list.append(row)

        if len(failed_list) != 0:
            failed_tab = pd.DataFrame(failed_list)
            failed_tab.to_csv("./failed_test_image_9c.csv")


tab = pd.read_csv("./data/test_preprocessed_tab.csv")
if __name__ == "__main__":
    while True:
        process_image(16)
