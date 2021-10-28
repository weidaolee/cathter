import os
import glob
import pandas as pd


def build_test_folder_tree():
    path_list = glob.glob("./data/test/*.jpg")
    path_list = [p.replace(".jpg", "") for p in path_list]

    for p in path_list:
        os.makedirs(p, exist_ok=True)

    tab = pd.DataFrame(path_list, columns=["Path"])
    tab["StudyInstanceUID"] = tab["Path"].apply(os.path.basename)

    tab.to_csv("./data/test_preprocessed_tab.csv", index=False)


def build_train_folder_tree():
    train_csv = pd.read_csv("./data/train.csv")
    base_dir = "./data/train"

    for uid in train_csv["StudyInstanceUID"].tolist():
        study_dir = os.path.join(base_dir, uid)
        os.makedirs(study_dir, exist_ok=True)


if __name__ == "__main__":
    build_train_folder_tree()
    build_test_folder_tree()
