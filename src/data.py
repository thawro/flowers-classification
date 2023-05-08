"""Dataset from https://www.robots.ox.ac.uk/~vgg/data/flowers/17/index.html"""

from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from typing import Callable
import numpy as np
from PIL import Image
from torchvision.transforms import Normalize


LABELS = [
    "Daffodil",
    "Snowdrop",
    "Lilly Valley",
    "Bluebell",
    "Crocus",
    "Iris",
    "Tigerlily",
    "Tulip",
    "Fritillary",
    "Sunflower",
    "Daisy",
    "Colts' Foot",
    "Dandelion",
    "Cowslip",
    "Buttercup",
    "Windflower",
    "Pansy",
]
NUM_CLASSES = len(LABELS)


def load_img_to_array(filepath) -> Image.Image:
    return Image.open(filepath)


class FlowersDataset(Dataset):
    def __init__(self, filepaths: np.ndarray, targets: np.ndarray, transform: Callable = None):
        self.filepaths = filepaths
        self.transform = transform
        self.targets = targets

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx: int):
        path = self.filepaths[idx]
        img = load_img_to_array(path)
        target = self.targets[idx]
        if self.transform is not None:
            img = self.transform(img)
        return img, target


class FlowersDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_ds: FlowersDataset,
        val_ds: FlowersDataset,
        test_ds: FlowersDataset,
        batch_size: int = 64,
        num_workers: int = 8,
    ):
        super().__init__()
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.test_ds = test_ds
        self.batch_size = batch_size
        self.num_workers = num_workers

    def train_dataloader(self):
        return DataLoader(
            self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers
        )

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(
            self.test_ds, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers
        )  # shuffle for plotting purposes only


MEAN_IMAGENET = [0.485, 0.456, 0.406]
STD_IMAGENET = [0.229, 0.224, 0.225]
new_mean = [-m / s for m, s in zip(MEAN_IMAGENET, STD_IMAGENET)]
new_std = [1 / s for s in STD_IMAGENET]


img_unnormalizer = Normalize(new_mean, new_std)
