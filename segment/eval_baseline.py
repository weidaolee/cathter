import pandas as pd
from sklearn.metrics import roc_auc_score

out_tab = pd.read_csv("./results/baseline/valid_output.csv")
gth_tab = pd.read_csv("./results/baseline/valid_target.csv")

gth_tab[["ETT", "NGT", "CVC"]] = 0

ett = [
    'ETT - Abnormal',
    'ETT - Borderline',
    'ETT - Normal',
]
ngt = [
    'NGT - Abnormal',
    'NGT - Borderline',
    'NGT - Normal',
    'NGT - Incompletely Imaged',
]

cvc = [
    'CVC - Abnormal',
    'CVC - Borderline',
    'CVC - Normal',
]

for k in ett:
    gth_tab["ETT"] += gth_tab[k]

for k in ngt:
    gth_tab["NGT"] += gth_tab[k]

for k in cvc:
    gth_tab["CVC"] += gth_tab[k]

out_tab[["ETT", "NGT", "CVC"]] = gth_tab[["ETT", "NGT", "CVC"]]

out_ett = out_tab[out_tab["ETT"] > 0]
gth_ett = gth_tab[out_tab["ETT"] > 0]

out_ngt = out_tab[out_tab["NGT"] > 0]
gth_ngt = gth_tab[out_tab["NGT"] > 0]

out_cvc = out_tab[out_tab["CVC"] > 0]
gth_cvc = gth_tab[out_tab["CVC"] > 0]

ett_auc = 0
for k in ett:
    auc = roc_auc_score(gth_ett[k], out_ett[k])
    print(k, auc)
    ett_auc += auc
print("ett auc", ett_auc / len(ett))
print("")

ngt_auc = 0
for k in ngt:
    auc = roc_auc_score(gth_ngt[k], out_ngt[k])
    print(k, auc)
    ngt_auc += auc
print("ngt auc", ngt_auc / len(ngt))
print("")

cvc_auc = 0
for k in cvc:
    auc = roc_auc_score(gth_cvc[k], out_cvc[k])
    print(k, auc)
    cvc_auc += auc
print("cvc auc", cvc_auc / len(cvc))
print("")
