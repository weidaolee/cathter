import pandas as pd

if __name__ == "__main__":

    train_csv = pd.read_csv("./data/train.csv")
    train_ann = pd.read_csv("./data/train_annotations.csv")

    keys = [
        'ETT - Abnormal',
        'ETT - Borderline',
        'ETT - Normal',
        'NGT - Abnormal',
        'NGT - Borderline',
        'NGT - Incompletely Imaged',
        'NGT - Normal',
        'CVC - Abnormal',
        'CVC - Borderline',
        'CVC - Normal',
        'Swan Ganz Catheter Present',
    ]

    # check weak label  balancedness
    csv_summary = train_csv[keys].sum()

    # check strong label balancedness
    ann_summary = train_ann["label"].value_counts()
