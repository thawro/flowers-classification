from torchmetrics import F1Score, Accuracy, MetricCollection
from typing import Dict, List
from torch import nn
import pytorch_lightning as pl
import torch
from typing import Literal
import wandb
from torch.nn.common_types import _size_2_t
from collections import OrderedDict
from data import NUM_CLASSES


class FlowersModule(pl.LightningModule):
    def __init__(
        self,
        lr: float = 0.01,
        weight_decay: float = 0.01,
    ):
        super().__init__()
        self.net = load_net()
        self.loss_fn = nn.NLLLoss(reduction="none")
        self.lr = lr
        self.weight_decay = weight_decay
        self.save_hyperparameters(ignore=["net"])
        self.outputs = {split: [] for split in ["train", "val", "test"]}
        self.examples = {split: {} for split in ["train", "val", "test"]}
        self.logged_metrics = {}

        metrics = MetricCollection(
            {
                "fscore": F1Score(task="multiclass", num_classes=17, average="weighted"),
                "accuracy": Accuracy(task="multiclass", num_classes=17, average="weighted"),
            }
        )
        self.train_metrics = metrics.clone(prefix=f"train/")
        self.val_metrics = metrics.clone(prefix=f"val/")
        self.test_metrics = metrics.clone(prefix=f"test/")
        self.metrics = {
            "train": self.train_metrics,
            "val": self.val_metrics,
            "test": self.test_metrics,
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def _produce_outputs(self, images: torch.Tensor, targets: torch.Tensor) -> Dict:
        log_probs = self(images)
        probs = torch.exp(log_probs)
        loss = self.loss_fn(log_probs, targets)
        preds = log_probs.argmax(dim=1)
        return {"loss": loss, "probs": probs, "preds": preds}

    def _common_step(
        self,
        batch: torch.Tensor,
        batch_idx: int,
        stage: Literal["train", "val", "test"],
    ) -> torch.Tensor:
        images, targets = batch
        outputs = self._produce_outputs(images, targets)
        outputs["targets"] = targets
        if stage == "test" and batch_idx == 0:
            examples = {"images": images, "targets": targets}
            examples.update(outputs)
            self.examples[stage] = {k: v.cpu() for k, v in examples.items()}
            del examples
        self.metrics[stage].update(outputs["probs"], targets)
        self.outputs[stage].append(outputs)
        return outputs["loss"].mean()

    def training_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, stage="train")

    def validation_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, stage="val")

    def test_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, stage="test")

    def _common_epoch_end(self, stage: Literal["train", "val", "test"]):
        outputs = self.outputs[stage]
        loss = torch.concat([output["loss"] for output in outputs]).mean().item()
        metrics = self.metrics[stage].compute()
        if self.trainer.sanity_checking:
            return loss
        loss_name = f"{stage}/loss"
        self.log(loss_name, loss, on_step=False, on_epoch=True, prog_bar=True, logger=False)
        self.logged_metrics.update({k: v.item() for k, v in metrics.items()})
        self.logged_metrics[loss_name] = loss
        wandb.log(self.logged_metrics, step=self.current_epoch)
        outputs.clear()
        self.logged_metrics.clear()
        self.metrics[stage].reset()

    def on_train_epoch_end(self):
        self._common_epoch_end("train")

    def on_validation_epoch_end(self):
        self._common_epoch_end("val")

    def on_test_epoch_end(self):
        self._common_epoch_end("test")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            params=self.parameters(),
            lr=self.lr,
            betas=(0.9, 0.999),
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.7,
            patience=7,
            threshold=0.0001,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }


"""SqueezeNet architecture based on https://arxiv.org/pdf/1602.07360.pdf.
By default, the simple bypass version is used
Also, BatchNormalization is added for each squeeze and expand layers"""


class CNNBlock(nn.Module):
    """Single CNN block constructed of combination of Conv2d, Activation, Pooling, Batch Normalization and Dropout."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: _size_2_t = 0,
        groups: int = 1,
        pool_kernel_size: _size_2_t = 1,
        pool_type: Literal["Max", "Avg"] = "Max",
        activation="ReLU",
    ):
        """
        Args:
            in_channels (int): Number of Conv2d input channels.
            out_channels (int): Number of Conv2d out channels.
            kernel_size (int): Conv2d kernel equal to `(kernel_size, kernel_size)`.
            stride (int, optional): Conv2d stride equal to `(stride, stride)`.
                Defaults to 1.
            padding (int | str, optional): Conv2d padding equal to `(padding, padding)`.
                Defaults to 1.. Defaults to 0.
            pool_kernel_size (int, optional): Pooling kernel equal to `(pool_kernel_size, pool_kernel_size)`.
                 Defaults to 1.
            pool_type (Literal["Max", "Avg"], optional): Pooling type. Defaults to "Max".
            use_batch_norm (bool, optional): Whether to use Batch Normalization (BN) after activation. Defaults to True.
            activation (str, optional): Type of activation function used before BN. Defaults to 0.
        """
        super().__init__()
        self.groups = groups
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.activation = activation
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            groups=groups,
            bias=False,
        )
        self.batch_norm = nn.BatchNorm2d(out_channels)

        self.activation_fn = getattr(nn, activation)()
        self.pool = getattr(nn, f"{pool_type}Pool2d")(pool_kernel_size, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        out = self.batch_norm(out)
        out = self.activation_fn(out)
        out = self.pool(out)
        return out


class DeepCNN(nn.Module):
    """Deep Convolutional Neural Network (CNN) constructed of many CNN blocks and ended with Global Average Pooling."""

    def __init__(
        self,
        in_channels: int,
        out_channels: List[int],
        kernels: List[_size_2_t],
        pool_kernels: List[_size_2_t],
        pool_type: Literal["Max", "Avg"] = "Max",
        activation: str = "ReLU",
    ):
        """
        Args:
            in_channels (int): Number of image channels.
            out_channels (list[int]): Number of channels used in CNN blocks.
            kernels (int | list[int]): Kernels of Conv2d in CNN blocks.
                If int or tuple[int, int] is passed, then all layers use same kernel size.
            pool_kernels (int | list[int]): Kernels of Pooling in CNN blocks.
                If int is passed, then all layers use same pool kernel size.
            pool_type (Literal["Max", "Avg"], optional): Pooling type in CNN blocks. Defaults to "Max".
            activation (str, optional): Type of activation function used in CNN blocks. Defaults to 0.
        """
        super().__init__()
        self.out_channels = out_channels
        self.kernels = kernels
        self.pool_kernels = pool_kernels
        self.pool_type = pool_type
        self.activation = activation
        n_blocks = len(out_channels)
        fixed_params = dict(
            pool_type=pool_type,
            activation=activation,
        )
        if isinstance(kernels, int) or isinstance(kernels, tuple):
            kernels = [kernels] * n_blocks
        if isinstance(pool_kernels, int) or isinstance(pool_kernels, tuple):
            pool_kernels = [pool_kernels] * n_blocks
        layers: list[tuple[str, nn.Module]] = [
            (
                f"conv_{i}",
                CNNBlock(
                    in_channels if i == 0 else out_channels[i - 1],
                    out_channels[i],
                    kernels[i],
                    pool_kernel_size=pool_kernels[i],
                    **fixed_params,
                ),
            )
            for i in range(n_blocks)
        ]
        self.net = nn.Sequential(OrderedDict(layers))
        self.out_channels = self.out_channels[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    @property
    def name(self):
        return "DeepCNN"


def load_net() -> nn.Module:
    backbone_net = DeepCNN(
        in_channels=3,
        out_channels=[16, 32, 64, 128],
        kernels=[3, 3, 3, 3],
        pool_kernels=[2, 2, 2, 2],
    )

    return nn.Sequential(
        backbone_net,
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(1, -1),
        nn.Linear(backbone_net.out_channels, NUM_CLASSES),
        nn.LogSoftmax(dim=1),
    )
