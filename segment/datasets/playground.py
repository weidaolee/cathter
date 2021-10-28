import os
import glob
import random

import numpy as np
import pandas as pd

df = pd.read_csv(
    "./data/train_seg_tab.csv",
    # usecols=[
    #     'StudyInstanceUID',
    #     "PatientID",
    #     "CVC",
    #     "CVC - Abnormal",
    #     "CVC - Borderline",
    #     # "NGT - Incompletely Imaged",
    #     "CVC - Normal",
    #     "Path",
    # ],
)

ab = df[df["NGT - Abnormal"] == 1]
ab = pd.concat([ab] * 20)

bord = df[df["NGT - Borderline"] == 1]
bord = pd.concat([bord] * 10)

df = pd.concat([df, ab, bord])
# df.to_csv("./data/train_seg_tab.csv", index=False)



# df = df[df["ETT"] == 1]


# df.to_csv("./data/valid_seg_tab.csv", index=False)

# df.to_csv("./data/train_seg_tab.csv", index=False)

# def load_image(row):
#     path = os.path.join(row["Path"], "image_9c_512.npz")
#     image = np.load(path, allow_pickle=True)["arr"].astype(np.uint8)
#     return image

# def load_mask(row):
#     path = os.path.join(row["Path"], "merged_label.npz")
#     mask = np.load(path, allow_pickle=True)["arr"].astype(np.float32)
#     return mask

# image_pool = {}
# mask_pool = {}

# for row in df.iterrows():
#     if len(image_pool) < 10:
#         row = row[1]
#         image_pool[row["Path"]] = load_image(row)
#         mask_pool[row["Path"]] = load_mask(row)

# k = random.choice(list(image_pool.keys()))
