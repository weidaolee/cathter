import os
import json
import torch

import tqdm
import torch.nn as nn
# import segmentation_models_pytorch as smp

from models.architectures import Unet
from models.components import Dense, ClassificationHeadOutput
from models.act_strategy import RReLUStrategy, PReLUStrategy

from prettytable import PrettyTable

from torch.utils.data import DataLoader
from datasets.segment import SegmentDataset


class SMPModel(Unet):
    """
    Based on segmentation models pytorch Unet.
    """
    def __init__(self, cfg):
        self.cfg = cfg

        super(SMPModel, self).__init__(**cfg["parameters"])

        self.prepare_act_strategy()

        self.prepare_encoder()
        self.prepare_decoder()

        self.prepare_classification_head()

        print("encoder conv1: ", self.encoder.conv1[:3])
        print("decoder conv1: ", self.decoder.blocks[0].conv1)
        print("classification_head: ", self.classification_head)

    def prepare_act_strategy(self):
        if "ReLU" == self.cfg["act_layer"]:
            raise NotImplementedError

        elif "RReLU" == self.cfg["act_layer"]:
            self.act_strategy = RReLUStrategy(
                self,
                lower=1 / 8,
                upper=1 / 3,
            )

        elif "PReLU" == self.cfg["act_layer"]:
            self.act_strategy = PReLUStrategy(
                self,
                encoder_init=1e-4,
                decoder_init=1e-2,
            )

        elif "SELU" == self.cfg["act_layer"]:
            raise NotImplementedError

        elif "CELU" == self.cfg["act_layer"]:
            raise NotADirectoryError

    def prepare_encoder(self):
        self.act_strategy.prepare_encoder()

    def prepare_decoder(self):
        self.act_strategy.prepare_decoder()

    def prepare_classification_head(self):
        head_in = self.encoder.out_channels[-1]
        cls_out = self.cfg["cls_out_features"]
        self.classification_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
            Dense(head_in, 512),
            Dense(512, 128),
            Dense(128, 32),
            ClassificationHeadOutput(32, cls_out),
        )


class SegmentModel(SMPModel):
    """
    Support dict multi-output.
    """
    def __init__(self, cfg):
        super(SegmentModel, self).__init__(cfg)

    def forward(self, x):
        x = x["image"]
        seg, cls = super().forward(x)
        seg = torch.sigmoid(seg)
        return {"seg": seg, "cls": cls}


class ClassifyModel(SMPModel):
    """
    Only contains encoder and classification_head.
    """
    def __init__(self, cfg):
        super(ClassifyModel, self).__init__(cfg)
        del self.decoder

        # self.encoder_list = self.encoder.get_stages()

    def forward(self, x):
        if "pred" in x:
            x = torch.cat([x["pred"], x["image"]], axis=1)
        else:
            x = x["image"]
        x = self.encoder(x)
        x = x[-1]
        x = self.classification_head(x)
        x = {"cls": x}
        return x


def compute_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params += param
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    with open("./task/stage1_cls_seg.json", "r") as f:
        cfg = json.load(f)
    print(cfg["model"])
    device = "cuda"
    device_ids = [int(x) for x in "0".split(",")]

    # check without decoder
    cfg["model"]["decoder"] = False
    model = ClassifyModel(cfg["model"])

    # model.encoder = torch.nn.DataParallel(
    #     model.encoder,
    #     device_ids=device_ids,
    # )
    # model.decoder = torch.nn.DataParallel(
    #     model.decoder,
    #     device_ids=device_ids,
    # )

    # model.classification_head = torch.nn.DataParallel(
    #     model.classification_head,
    #     device_ids=device_ids,
    # )
    # model.encoder = torch.nn.DataParallel(model.encoder, device_ids=[0])

    # model.classification_head = torch.nn.DataParallel(
    #     model.classification_head, device_ids=[0])

    # model = model.to(device)
    # state_dict = torch.load("./results/batch_norm_relu/checkpoints/model.pth")

    # model.load_state_dict(state_dict["model"], strict=False)

    # dataset = SegmentDataset("./data/train_seg_tab.csv", **cfg["dataset"])
    # dataloader = DataLoader(dataset,
    #                         batch_size=4,
    #                         shuffle=False,
    #                         num_workers=36,
    #                         drop_last=True,
    #                         prefetch_factor=1)

    # for _, inputs, target in tqdm.tqdm(dataloader, total=len(dataloader)):
    #     break

    # gpu_id = 0
    # x = image.to(f"{device}:{gpu_id}")
    # # image = image.to(gpu_id)
    # # mask = mask.to(f"{device}:{gpu_id}")

