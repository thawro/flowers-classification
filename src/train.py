from pathlib import Path
from typing import List, Literal, Dict
import scipy.io
import numpy as np
from torch import nn
import torchvision
from pytorch_lightning import seed_everything
import pytorch_lightning as pl
import torchvision.transforms as T
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.callbacks import RichProgressBar, EarlyStopping
from data import FlowersDataModule, FlowersDataset
from model import FlowersModule

DATA_PATH = Path("data/jpg")
FILENAMES_PATH = DATA_PATH / "files.txt"

MEAN_IMAGENET = [0.485, 0.456, 0.406]
STD_IMAGENET = [0.229, 0.224, 0.225]

datasplits = scipy.io.loadmat("datasplits.mat")


def load_filenames(filepath: Path) -> List[str]:
    with open(filepath) as f:
        filenames = f.readlines()
        return [filename.strip() for filename in filenames]


def load_idxs_from_datasplits(datasplits: Dict, split_idx: Literal[1, 2, 3]):
    train_idxs = datasplits[f"trn{split_idx}"][0] - 1
    val_idxs = datasplits[f"val{split_idx}"][0] - 1
    test_idxs = datasplits[f"tst{split_idx}"][0] - 1
    return train_idxs, val_idxs, test_idxs


if __name__ == "__main__":
    seed_everything(42)

    filenames = load_filenames(FILENAMES_PATH)
    filepaths = np.array([str(DATA_PATH / name) for name in filenames])
    targets = np.arange(1360) // 80

    train_idxs, val_idxs, test_idxs = load_idxs_from_datasplits(datasplits, split_idx=1)

    train_transform = T.Compose(
        [
            T.ToTensor(),
            T.Normalize(MEAN_IMAGENET, STD_IMAGENET),
            T.Resize(256),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomCrop(224),
        ]
    )

    inference_transform = T.Compose(
        [
            T.ToTensor(),
            T.Normalize(MEAN_IMAGENET, STD_IMAGENET),
            T.Resize(256),
            T.CenterCrop(224),
        ]
    )

    train_filepaths, train_targets = filepaths[train_idxs], targets[train_idxs]
    val_filepaths, val_targets = filepaths[val_idxs], targets[val_idxs]
    test_filepaths, test_targets = filepaths[test_idxs], targets[test_idxs]

    train_ds = FlowersDataset(
        filepaths=train_filepaths, transform=train_transform, targets=train_targets
    )
    val_ds = FlowersDataset(
        filepaths=val_filepaths, transform=inference_transform, targets=val_targets
    )
    test_ds = FlowersDataset(
        filepaths=test_filepaths, transform=inference_transform, targets=test_targets
    )

    datamodule = FlowersDataModule(train_ds, val_ds, test_ds, batch_size=32)

    net = torchvision.models.resnet18(pretrained=True)
    net.fc = nn.Linear(512, 17)

    model = FlowersModule()

    callbacks = [RichProgressBar(), EarlyStopping(monitor="val/loss", patience=15)]

    logger = WandbLogger(name="test", project="flowers-classification")

    trainer = pl.Trainer(
        accelerator="auto", callbacks=callbacks, logger=logger, log_every_n_steps=20
    )

    trainer.fit(model, datamodule)
