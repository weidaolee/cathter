import os
import glob
import tqdm

import multiprocessing

import numpy as np
import pandas as pd

ann = pd.read_csv("./data/train_preprocessed_ann.csv")

dir_list = list(set(ann["Path"]))
# dir_list = pd.read_csv("./failed_label_path.csv", header=None)
# dir_list = list(dir_list[0])

size = 512

# def merge_gth(dir_path):
#     try:
#         path_list = glob.glob(os.path.join(dir_path, "label_*.npz"))

#         arr = np.zeros([size, size, 1], dtype=bool)

#         for p in path_list:
#             arr = arr + np.load(p)["label"].astype(bool)

#         arr = arr.astype(bool)
#         arr = arr.astype(bool)
#         save_path = os.path.join(dir_path, "merged_label.npz")
#         np.savez_compressed(save_path, label=arr)
#         return None

#     except:
#         return dir_path


def merge_gth(dir_path):

    try:
        path_list = glob.glob(os.path.join(dir_path, "label_*.npz"))

        arr = np.zeros([size, size], dtype=bool)

        for p in path_list:
            arr = arr + np.load(p, allow_pickle=True)["label"]

        arr = arr.astype(bool)

        if len(arr.shape) != 2:
            print(dir_path)
            return dir_path

        save_path = os.path.join(dir_path, "merged_label.npz")
        np.savez_compressed(save_path, label=arr)
        return None

    except:
        print(dir_path)
        return dir_path


def process_label():
    failed_path = []
    with multiprocessing.Pool(32) as pool:
        gen = pool.imap(merge_gth, dir_list)
        for p in tqdm.tqdm(gen, total=len(dir_list)):
            if p is not None:
                print(p)
                failed_path.append(p)

    df = pd.DataFrame(data=failed_path)
    df.to_csv("./failed_label_path.csv", index=False, header=None)


if __name__ == "__main__":
    process_label()
