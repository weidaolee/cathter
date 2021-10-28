import pandas as pd


def catheter_exist(row, type):
    findings = []
    for f in row.index:
        if type in f:
            findings.append(f)

    row = row[findings]

    row = pd.to_numeric(row)

    if row.sum() == 0:
        return 0
    else:
        return 1


def process_train_tab():
    tab = pd.read_csv("./data/train_tab.csv")

    for t in ["ETT", "NGT", "CVC"]:
        tab[t] = tab.apply(lambda row: catheter_exist(row, t), axis=1)

    tab.to_csv("./data/train_tab.csv", index=False)


def process_valid_tab():
    tab = pd.read_csv("./data/valid_tab.csv")

    for t in ["CVC", "ETT", "NGT"]:
        tab[t] = tab.apply(lambda row: catheter_exist(row, t), axis=1)

    tab.to_csv("./data/valid_tab.csv", index=False)


if __name__ == "__main__":
    process_train_tab()
    process_valid_tab()
