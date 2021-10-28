import pandas as pd


sample = pd.read_csv("./data/sample_submission.csv")
sample["Path"] = sample["StudyInstanceUID"].apply(lambda s: "./data/test/" + s)
sample.to_csv("./data/submission.csv")
