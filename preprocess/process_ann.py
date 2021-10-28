import os
import json
import tqdm
import multiprocessing

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from scipy.interpolate import interp1d


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


def _swap_to_yx(row):
    seg = eval(row["data"])
    seg = [[p[1], p[0]] for p in seg]

    seg = json.dumps(seg)
    return seg


# def _pad(row):
#     seg = row["data"]
#     shift_y = row["Shift Y"]
#     shift_x = row["Shift X"]

#     seg = np.array(eval(seg))

#     seg[:, 0] += shift_y
#     seg[:, 1] += shift_x

#     seg = json.dumps(seg.tolist())
#     return seg


class Interpolation:
    def __init__(self):
        ...

    def __call__(self, row):
        seg = eval(row["data"])
        seg = self.check_no_duplicates(seg)
        seg = self.interpolate(seg)

        seg = json.dumps(seg)

        return seg

    def interpolate(self, seg: list) -> list:
        seg = np.array(seg)

        l = np.linalg.norm(seg[1:] - seg[:-1], axis=-1)

        t = np.cumsum(l)

        t = np.concatenate([[0], t])
        t2 = np.arange(t[-1])

        y, x = seg[:, 0], seg[:, 1]

        y2 = interp1d(t,
                      y,
                      kind=self.choose_kind(seg),
                      bounds_error=False,
                      fill_value="extrapolate")(t2)

        x2 = interp1d(t,
                      x,
                      kind=self.choose_kind(seg),
                      bounds_error=False,
                      fill_value="extrapolate")(t2)

        seg = np.stack([y2, x2], axis=-1).astype(int)

        seg = seg.tolist()

        return seg

    def check_no_duplicates(self, seg: list) -> list:
        res = [seg[0]]

        for i in range(1, len(seg)):
            if seg[i] != seg[i - 1]:
                res.append(seg[i])

        return res

    def choose_kind(self, seg: list) -> str:
        if len(seg) < 4:
            return "linear"
        else:
            return "cubic"


if __name__ == "__main__":
    path_base = "./data/train"
    ann = pd.read_csv("./data/train_annotations.csv")

    uid_series = ann["StudyInstanceUID"]
    ann["Path"] = uid_series.apply(lambda x: os.path.join(path_base, x))
    path_list = list(set(ann["Path"].tolist()))

    shape_list = get_ori_shape(path_list)
    shape_map = dict(zip(path_list, shape_list))

    ann["Original Y"] = ann["Path"].apply(lambda p: shape_map[p][0])
    ann["Original X"] = ann["Path"].apply(lambda p: shape_map[p][1])

    # ann["Shift Y"] = ann["Original Y"].apply(lambda y: (3827 - y) // 2)
    # ann["Shift X"] = ann["Original X"].apply(lambda x: (3567 - x) // 2)

    ann["data"] = ann.apply(lambda row: _swap_to_yx(row), axis=1)

    interp = Interpolation()

    # ann["Padded Coordinates"] = ann.apply(lambda row: _pad(row), axis=1)
    ann["Interpolated Points"] = ann.apply(lambda row: interp(row), axis=1)

    ann = ann[[
        'StudyInstanceUID',
        'label',
        'data',
        'Original Y',
        'Original X',
        # 'Shift Y',
        # 'Shift X',
        'Interpolated Points',
        'Path',
    ]]

    ann.to_csv("./data/train_preprocessed_ann.csv", index=False)

    print("finished!!")
