import json
import argparse

from datasets.segment import SegmentDataset
from datasets.classify import SimpleClassifyDataset
from datasets.classify import SegmentBasedClassifyDataset

from utils.trainer import SegmentTrainer
from utils.trainer import ClassifyTrainer

from losses.base import BaseLoss
from losses.dice import DiceLoss
from losses.iou import IoULoss
from losses.lovasz import LovaszLoss
from losses.focal import BinaryFocalLoss
from losses.bce import SegmentBCELoss
from losses.bce import ClassifyBCELoss
from losses.ce import CrossEntropyLoss

from models.smp import SegmentModel
from models.smp import ClassifyModel


def train(Model, Dataset, Trainer):

    with open(args.config_path) as f:
        cfg = json.load(f)

    model = get_model(cfg, Model)
    loss = get_loss(cfg)
    train_set, valid_set = get_dataset(args, cfg, Dataset)

    Trainer(
        args=args,
        cfg=cfg,
        loss=loss,
        model=model,
        train_set=train_set,
        valid_set=valid_set,
    )


def get_loss(cfg):
    print("Prepare criterion...")

    cfg = cfg["loss"]
    criterion = BaseLoss()
    if "dice" in cfg:
        loss = "dice"
        criterion = DiceLoss(
            base_loss=criterion,
            **cfg[loss],
        )
        print(f"  consider {criterion.name}...")

    if "iou" in cfg:
        loss = "iou"
        criterion = IoULoss(
            base_loss=criterion,
            **cfg[loss],
        )
        print(f"  consider {criterion.name}...")

    if "lovasz" in cfg:
        loss = "lovasz"
        criterion = LovaszLoss(
            base_loss=criterion,
            **cfg[loss],
        )
        print(f"  consider {criterion.name}...")

    if "focal" in cfg:
        loss = "focal"
        criterion = BinaryFocalLoss(
            base_loss=criterion,
            **cfg[loss],
        )
        print(f"  consider {criterion.name}...")

    if "seg_bce" in cfg:
        loss = "seg_bce"
        criterion = SegmentBCELoss(
            base_loss=criterion,
            **cfg[loss],
        )
        print(f"  Consider {criterion.name}...")

    if "cls_bce" in cfg:
        loss = "cls_bce"

        for k in cfg[loss]:
            print(cfg[loss])
            criterion = ClassifyBCELoss(
                key=k,
                base_loss=criterion,
                **cfg[loss][k],
            )
        print(f"  Consider {criterion.name}.")

    if "ce" in cfg.keys():
        loss = "ce"

        for k in cfg[loss]:
            print(cfg[loss])
            criterion = CrossEntropyLoss(
                key=k,
                base_loss=criterion,
                **cfg[loss][k],
            )

    return criterion


def get_model(cfg, Model):
    print(cfg["model"])
    model = Model(cfg["model"])
    return model


def get_dataset(args, cfg, Dataset):
    print("Prepare dataset...")
    print(cfg["dataset"])

    train_path = args.train_path
    valid_path = args.valid_path

    train_set = Dataset(
        train_path,
        **cfg["dataset"],
        is_train=True,
    )
    valid_set = Dataset(
        valid_path,
        **cfg["dataset"],
        is_train=False,
    )
    return train_set, valid_set


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task_type",
        type=str,
    )
    parser.add_argument(
        "--task_name",
        type=str,
    )
    parser.add_argument(
        '--gpus',
        type=str,
        help='specify which GPUs to use.',
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=32,
        help='number of workers',
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='pytorch checkpoint file path',
    )
    parser.add_argument(
        "--config_path",
        type=str,
    )
    parser.add_argument(
        "--train_path",
        type=str,
    )
    parser.add_argument(
        "--valid_path",
        type=str,
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(args)

    if args.task_type == "segmentation with catheter appearence":
        train(
            Model=SegmentModel,
            Dataset=SegmentDataset,
            Trainer=SegmentTrainer,
        )
    elif args.task_type == "classification with catheter seg input":
        train(
            Model=ClassifyModel,
            Dataset=SegmentBasedClassifyDataset,
            Trainer=SegmentTrainer,
        )

    elif args.task_type == "classification with catheter seg model":
        train(
            Model=SegmentModel,
            Dataset=SegmentDataset,
            Trainer=ClassifyTrainer,
        )

    elif args.task_type == "simple classification":
        train(
            Model=ClassifyModel,
            Dataset=SimpleClassifyDataset,
            Trainer=ClassifyTrainer,
        )
