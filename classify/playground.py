import glob
import pandas as pd

path_list = glob.glob("./data/test/*.jpg")


res = pd.read_csv("./results/overall/checkpoints/model_024.csv")
del res["Path"]


res.to_csv("overall_024.csv", index=False)
