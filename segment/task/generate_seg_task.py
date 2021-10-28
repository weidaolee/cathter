import json
import os
import glob

add_path = set()
# for p in glob.glob("../*/*.py"):
#     add_path.add(os.path.dirname(os.path.abspath(p)))

# for p in glob.glob("../*.py"):
#     add_path.add(os.path.dirname(os.path.abspath(p)))

add_path = list(add_path)

export_path_string = f"export PYTHONPATH={(os.path.abspath('..'))}\n"
# for p in add_path:
#     export_path_string += f"export PYTHONPATH=$PYTHONPATH:{p}\n"


def search_encoder():
    with open("../config.json") as f:
        cfg = json.load(f)

    encoder_list = [
        "resnet101",  # 42
        "resnext101_32x4d",  # 42
        "timm-resnest101e",  # 46
        "timm-res2net101_26w_4s",  # 43
        "timm-res2net50_26w_8s",  # 46
        "se_resnet101",  # 47
        "se_resnext101_32x4d",  # 46
    ]

    for e in encoder_list:
        cfg["train"]["batch_size"] = 64
        cfg["model"]["parameters"]["encoder_name"] = e
        if e == "resnext101_32x4d":
            cfg["model"]["parameters"]["encoder_weights"] = "swsl"
        else:
            cfg["model"]["parameters"]["encoder_weights"] = "imagenet"

        s = json.dumps(cfg, indent=4)
        with open(f"encoder_{e}.json", "w") as f:
            f.write(s)

        s = f'{export_path_string} python train.py \\\n\t   --prefix="encoder_{e}" \\\n\t   --gpu="0,1,2,3,4,5,6,7" \\\n\t   --config_path="./task/encoder_{e}.json" \\\n\t   --train_path="./data/train_seg_tab.csv" \\\n\t   --valid_path="./data/valid_seg_tab.csv" \\\n'

        with open(f"encoder_{e}.sh", "w") as f:
            f.write(s)


def search_architecture():
    with open("../config.json") as f:
        cfg = json.load(f)

    architecture_list = [
        "Unet++",
        "Linknet",
        "FPN",
        "PSPNet",
        "PAN",
        "DeepLabV3",
        "DeepLabV3Plus",
    ]

    for a in architecture_list:
        cfg["train"]["batch_size"] = 64
        cfg["model"]["architecture"] = a
        cfg["model"]["parameters"]["encoder_name"] = "timm-resnest101e"

        a = a.replace("++", "PlusPlus")
        s = json.dumps(cfg, indent=4)
        with open(f"architecture_{a}.json", "w") as f:
            f.write(s)

        s = f'{export_path_string} python train.py \\\n\t   --prefix="architecture_{a}" \\\n\t   --gpu="0,1,2,3,4,5,6,7" \\\n\t   --config_path="./task/architecture_{a}.json" \\\n\t   --train_path="./data/train_seg_tab.csv" \\\n\t   --valid_path="./data/valid_seg_tab.csv" \\\n'

        with open(f"architecture_{a}.sh", "w") as f:
            f.write(s)


if __name__ == "__main__":
    search_encoder()
    search_architecture()
