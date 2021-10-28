import os
import pandas as pd
import json

train_csv = pd.read_csv("./data/train.csv")
train_ann = pd.read_csv("./data/train_annotations.csv")


class ParseInfoToJson:
    def __init__(self):
        self.json_info = None
        self.tags = [
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

        self.parse_csv_info()
        self.parse_ann_info()
        self.parse_uid_to_path()

    def save_to_json(self, path):
        json_str = json.dumps(self.json_info, indent=4)
        with open(path, 'w') as f:
            f.write(json_str)

    def parse_csv_info(self):
        def _csv_to_dict(row):
            row = row[1]
            uid = row["StudyInstanceUID"]
            pid = row["PatientID"]
            tags = {t: int(row[t]) for t in self.tags}

            tags["StudyInstanceUID"] = uid
            tags["PatientID"] = pid

            return uid, tags

        self.json_info = dict([_csv_to_dict(r) for r in train_csv.iterrows()])

    def parse_ann_info(self):
        ann_json = {k: {} for k in train_ann["StudyInstanceUID"].tolist()}

        def _ann_to_dict(row):
            row = row[1]
            uid = row["StudyInstanceUID"]
            tag = row["label"]
            target = eval(row["data"])
            if tag not in ann_json[uid]:
                ann_json[uid][tag] = [target]

            else:
                ann_json[uid][tag].append(target)

        [_ann_to_dict(r) for r in train_ann.iterrows()]

        for uid in ann_json.keys():
            self.json_info[uid]["Segments"] = ann_json[uid]

    def parse_uid_to_path(self):
        json_info = {
            os.path.join("./data/train", uid): self.json_info[uid]
            for uid in self.json_info.keys()
        }

        self.json_info = json_info


if __name__ == "__main__":
    parser = ParseInfoToJson()
    parser.save_to_json("./data/datalist/train.json")
