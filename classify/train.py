import os
import argparse
from models.resnet200 import ResNet200D

from utils.trainer import Trainer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prefix",
        type=str,
    )
    parser.add_argument(
        '--gpu',
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
        "--train_path",
        type=str,
    )
    parser.add_argument(
        "--valid_path",
        type=str,
    )
    return parser.parse_args()


def train_overall():
    from datasets.overall import OverallDataset
    from losses.overall import OverallBCELoss
    from utils.logger import OverallValidLogger

    model = ResNet200D(out_dim=11)
    dataset = OverallDataset
    criterion = OverallBCELoss()
    valid_logger = OverallValidLogger

    trainer = Trainer(
        args=args,
        model=model,
        dataset=dataset,
        criterion=criterion,
        valid_logger=valid_logger,
        load_ckpt=False,
    )

    trainer.train()


def train_exist():
    from datasets.exist import ExistDataset
    from losses.exist import ExistBCELoss
    from utils.logger import ExistLogger

    model = ResNet200D(out_dim=4)
    dataset = ExistDataset
    criterion = ExistBCELoss()
    valid_logger = ExistLogger

    trainer = Trainer(
        args=args,
        model=model,
        dataset=dataset,
        criterion=criterion,
        valid_logger=valid_logger,
        load_ckpt=False,
    )

    trainer.train()


if __name__ == "__main__":
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    if args.prefix == "overall":
        train_overall()
    if args.prefix == "exist":
        train_exist()
