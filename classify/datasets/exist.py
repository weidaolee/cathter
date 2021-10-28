import pandas as pd

from datasets.overall import OverallDataset
# from overall import OverallDataset


class ExistDataset(OverallDataset):
    def __init__(self, csv_path, is_train=False):
        super(ExistDataset, self).__init__(csv_path, is_train)
        self.df = pd.read_csv(csv_path)

        self.is_train = is_train

        self.categories = [
            'ETT',
            'NGT',
            'CVC',
            'Swan Ganz Catheter Present',
        ]


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    dataset = ExistDataset("./data/train_tab.csv", is_train=True)

    targets = []
    dataloader = DataLoader(dataset,
                            batch_size=128,
                            shuffle=False,
                            num_workers=1,
                            prefetch_factor=1)

    for i, batch in enumerate(dataloader):

        targets.append(batch[1].numpy())
        break

    # targets = np.concatenate(targets, axis=0)
