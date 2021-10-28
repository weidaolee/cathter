import tqdm
import pandas as pd

if __name__ == "__main__":
    ann = pd.read_csv("./data/train_preprocessed_ann.csv")
    new_ann = pd.read_csv("./preprocess/kaggle_represent_list.csv")

    def correct_label(i, label):
        ann.iloc[i]["label"] = label

    for row in tqdm.tqdm(new_ann.iterrows(), total=new_ann.shape[0]):
        row = row[1]

        i = row["index"]
        label = row["new_label"]

        correct_label(i, label)

    ann.to_csv("./data/train_preprocessed_ann.csv", index=False)
