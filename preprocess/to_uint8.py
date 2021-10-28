import os
import tqdm

import numpy as np
import pandas as pd

import multiprocessing


def to_uint8(row):
    try:
        row = row[1]
        path = os.path.join(row["Path"], "image_9c_512.npz")
        image = np.load(path, allow_pickle=True)["image"]

        for i in range(9):
            img = image[..., i]
            img = (img - img.min()) / (img.max() - img.min())
            img *= 255
            image[..., i] = img

        image = image.astype(np.uint8)
        np.savez_compressed(path, image=image)

        return None
    except:
        return row


if __name__ == "__main__":
    df = pd.read_csv("./data/train_preprocessed_tab.csv")
    failed_list = []
    with multiprocessing.Pool(36) as pool:
        gen = pool.imap(to_uint8, df.iterrows())
        for row in tqdm.tqdm(gen, total=df.shape[0]):
            if row is not None:
                failed_list.append(row)

    if len(failed_list) != 0:
        failed_tab = pd.DataFrame(failed_list)
        failed_tab.to_csv("./failed_to_uint8.csv")
