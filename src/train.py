from pathlib import Path
from typing import List, Literal, Dict
import scipy.io
import numpy as np
from pytorch_lightning import seed_everything
import pytorch_lightning as pl
import torchvision.transforms as T
import torch
import wandb
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.callbacks import RichProgressBar, EarlyStopping, ModelCheckpoint

from data import FlowersDataModule, FlowersDataset, MEAN_IMAGENET, STD_IMAGENET, LABELS
from model import FlowersModule
from callbacks import ConfusionMatrixLogger, ExamplePredictionsLogger


DATA_PATH = Path("data/jpg")
FILENAMES_PATH = DATA_PATH / "files.txt"

NUM_EXAMPLES_PER_CLASS = 80

MAX_EPOCHS = 150

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
    targets = np.arange(len(filepaths)) // NUM_EXAMPLES_PER_CLASS

    train_idxs, val_idxs, test_idxs = load_idxs_from_datasplits(datasplits, split_idx=1)

    train_transform = T.Compose(
        [
            T.Resize(256),
            T.RandomCrop(224),
            T.RandomHorizontalFlip(p=0.5),
            T.ToTensor(),
            T.Normalize(MEAN_IMAGENET, STD_IMAGENET),
        ]
    )

    inference_transform = T.Compose(
        [
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(MEAN_IMAGENET, STD_IMAGENET),
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

    datamodule = FlowersDataModule(train_ds, val_ds, test_ds, batch_size=64)

    model = FlowersModule()
    name = model.net[0].name
    model_path = f"models/{name}.pt"

    ckpt_callback = ModelCheckpoint(
        dirpath="ckpts", filename="best", monitor="val/loss", save_last=False, save_top_k=1
    )
    callbacks = [
        ckpt_callback,
        RichProgressBar(),
        EarlyStopping(monitor="val/loss", patience=15),
        ConfusionMatrixLogger(classes=LABELS),
        ExamplePredictionsLogger(classes=LABELS),
    ]

    logger = WandbLogger(name=name, project="flowers-classification")

    trainer = pl.Trainer(
        accelerator="auto",
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=20,
        max_epochs=MAX_EPOCHS,
        deterministic=True,
    )
    trainer.fit(model, datamodule)

    best_model = FlowersModule.load_from_checkpoint(ckpt_callback.best_model_path)
    model_scripted = torch.jit.script(best_model.net, torch.rand((1, 3, 224, 224)))
    model_scripted.save(model_path)
    trainer.validate(best_model, datamodule)
    trainer.test(best_model, datamodule)

    wandb.save(model_path)

    wandb.finish()
