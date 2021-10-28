import os
import tqdm

import multiprocessing

import pandas as pd
import matplotlib.pyplot as plt

from clahe import CLAHE


def _get_ori_shape(p):
    img = plt.imread(p + ".jpg")
    return img.shape


def get_ori_shape(path_list):
    shape_list = []
    with multiprocessing.Pool(64) as pool:
        gen = pool.imap(_get_ori_shape, path_list)
        for shape in tqdm.tqdm(gen, total=len(path_list)):
            shape_list.append(shape)

    return shape_list


def expand_table_info():
    tab = pd.read_csv("./data/train.csv")

    path_base = "./data/train"
    uid_series = tab["StudyInstanceUID"]
    tab["Path"] = uid_series.apply(lambda x: os.path.join(path_base, x))
    path_list = list(set(tab["Path"].tolist()))

    shape_list = get_ori_shape(path_list)
    shape_map = dict(zip(path_list, shape_list))

    tab["Original Y"] = tab["Path"].apply(lambda p: shape_map[p][0])
    tab["Original X"] = tab["Path"].apply(lambda p: shape_map[p][1])

    tab.to_csv("./data/train_preprocessed_tab.csv", index=False)


def process_image(worker=64):
    with multiprocessing.Pool(worker) as pool:
        failed_list = []
        gen = pool.imap(CLAHE(), tab.iterrows())

        for row in tqdm.tqdm(gen, total=tab.shape[0]):
            if row is not None:
                failed_list.append(row)

    if len(failed_list) > 0:
        failed_tab = pd.DataFrame([failed_list])
        failed_tab.to_csv("failed_process_img.csv")


failed_list = [
    48, 49, 385, 51, 52, 53, 54, 387, 55, 56, 389, 58, 59, 60, 61, 391, 62, 63,
    392, 393, 394, 395, 66, 67, 69, 70, 71, 399, 38, 405, 40, 42, 43, 44, 45,
    46, 47, 2243, 21864, 21865, 21866, 21867, 21868, 21869, 21870, 21871,
    21872, 21873, 21874, 21875, 21876, 21877, 21849, 21878, 21894, 21879,
    21880, 21881, 21882, 21855, 21858, 21859, 21860, 21861, 21862, 21863
]
tab = pd.read_csv("./data/train_preprocessed_tab.csv").iloc[failed_list]

if __name__ == "__main__":
    # expand_table_info()
    process_image(worker=35)
