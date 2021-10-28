import pandas as pd

tab = pd.read_csv("./data/train_tab.csv")


def catheter_exist(row, type):

    row = row[[
        f"{type} - Abnormal",
        f"{type} - Borderline",
        f"{type} - Normal",
    ]]

    row = pd.to_numeric(row)

    if row.sum() == 0:
        return 0
    else:
        return 1


for t in ["ETT", "NGT", "CVC"]:
    tab[t] = tab.apply(lambda row: catheter_exist(row, t), axis=1)
