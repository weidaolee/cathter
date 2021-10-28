import pandas as pd
if __name__ == "__name__":
    tab = pd.read_csv("./data/train_preprocessed_tab.csv")
    ann = pd.read_csv("./data/train_preprocessed_ann.csv")

    info = tab[[
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
    ]]

    valid_tab = tab.sample(frac=0.1)

    valid_info = valid_tab[[
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
    ]]

    print(info.sum() // 10)
    print(valid_info.sum())

    train_index = list(set(tab.index) - set(valid_tab.index))

    train_tab = tab.loc[train_index]
    train_info = train_tab[[
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
    ]]

    print(train_info.sum())

    def concat(base_table, category, times):
        cond = (base_table[category] == 1) & (base_table["CVC - Normal"] != 1)
        seq = [base_table[cond] for _ in range(times)]
        base_table = pd.concat([base_table] + seq)
        base_info = base_table[[
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
        ]]

        return base_table, base_info

    train_tab, train_info = concat(train_tab, "ETT - Abnormal", 100)
    train_tab, train_info = concat(train_tab, "ETT - Borderline", 5)
    train_tab, train_info = concat(train_tab, "NGT - Abnormal", 10)
    train_tab, train_info = concat(train_tab, "NGT - Borderline", 10)
    train_tab, train_info = concat(train_tab, "Swan Ganz Catheter Present", 10)

    print(train_info.sum())

    train_tab.to_csv("./data/train_tab.csv", index=False)
    valid_tab.to_csv("./data/valid_tab.csv", index=False)

    valid_uid_set = set(valid_tab["StudyInstanceUID"])

    def in_valid_set(uid):
        if uid in valid_uid_set:
            return "valid"
        else:
            return "train"

    ann["Dataset"] = ann["StudyInstanceUID"].apply(in_valid_set)

    train_ann = ann[ann["Dataset"] == "train"]
    valid_ann = ann[ann["Dataset"] == "valid"]
    del train_ann["Dataset"]
    del valid_ann["Dataset"]

    train_ann.to_csv("./data/train_ann.csv", index=False)
    valid_ann.to_csv("./data/valid_ann.csv", index=False)
