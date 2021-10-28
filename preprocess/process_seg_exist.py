import pandas as pd

train_ann = pd.read_csv("./data/train_ann.csv")
valid_ann = pd.read_csv("./data/valid_ann.csv")

train_tab = pd.read_csv("./data/train_tab.csv")
valid_tab = pd.read_csv("./data/valid_tab.csv")

train_ann_set = set(train_ann["StudyInstanceUID"])
valid_ann_set = set(valid_ann["StudyInstanceUID"])

train_tab_set = set(train_tab["StudyInstanceUID"])
valid_tab_set = set(valid_tab["StudyInstanceUID"])


def process_label(row, is_train):
    uid = row["StudyInstanceUID"]

    if is_train:
        ann = train_ann

    else:
        ann = valid_ann

    sub_ann = ann[ann["StudyInstanceUID"] == uid]

    def process(sub_ann_row):
        label = sub_ann_row["label"]
        row[label] += 1

    row.fillna(0, inplace=True)
    sub_ann.apply(process, axis=1)


def process_const_tag(row, is_train):
    uid = row["StudyInstanceUID"]

    if is_train:
        tab = train_tab
    else:
        tab = valid_tab

    sub_tab = tab[tab["StudyInstanceUID"] == uid].iloc[0]

    const_tags = [
        'PatientID',
        'Path',
        'Original Y',
        'Original X',
        'ETT',
        'NGT',
        'CVC',
    ]

    row[const_tags] = sub_tab[const_tags]


if __name__ == "__main__":
    train_seg = pd.DataFrame(columns=train_tab.columns)
    train_seg["StudyInstanceUID"] = list(train_ann_set)

    train_seg.apply(lambda r: process_label(row=r, is_train=True), axis=1)
    train_seg.apply(lambda r: process_const_tag(row=r, is_train=True), axis=1)
    train_seg.to_csv("./data/train_seg_tab.csv", index=False)

    valid_seg = pd.DataFrame(columns=valid_tab.columns)
    valid_seg["StudyInstanceUID"] = list(valid_ann_set)

    valid_seg.apply(lambda r: process_label(row=r, is_train=False), axis=1)
    valid_seg.apply(lambda r: process_const_tag(row=r, is_train=False), axis=1)
    valid_seg.to_csv("./data/valid_seg_tab.csv", index=False)
