from torchmetrics import F1Score, Accuracy, AUROC, MetricCollection
from typing import Dict, List
from torch import nn
import pytorch_lightning as pl
import torch
from typing import Literal
import wandb
from torch.nn.common_types import _size_2_t
from collections import OrderedDict


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
        use_batch_norm: bool = True,
        dropout: float = 0,
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
            dropout (float, optional): Dropout probability (used after BN). Defaults to 0.
            activation (str, optional): Type of activation function used before BN. Defaults to 0.
        """
        super().__init__()
        if isinstance(pool_kernel_size, int):
            self.use_pool = pool_kernel_size > 1
        else:
            self.use_pool = all(dim == 1 for dim in pool_kernel_size)
        self.use_batch_norm = use_batch_norm
        self.use_dropout = dropout > 0
        self.groups = groups
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.activation = activation
        use_bias = not use_batch_norm
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            groups=groups,
            bias=use_bias,
        )
        if self.use_batch_norm:
            self.batch_norm = nn.BatchNorm2d(out_channels)

        self.linear = activation is None
        if not self.linear:
            self.activation_fn = getattr(nn, activation)()
        if self.use_pool:
            self.pool = getattr(nn, f"{pool_type}Pool2d")(pool_kernel_size, stride=2)

        if self.use_dropout:
            self.dropout = nn.Dropout2d(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        if self.use_batch_norm:
            out = self.batch_norm(out)
        if not self.linear:
            out = self.activation_fn(out)
        if self.use_pool:
            out = self.pool(out)
        if self.use_dropout:
            out = self.dropout(out)
        return out


class FireBlock(nn.Module):
    """FireBlock used to squeeze and expand convolutional channels"""

    def __init__(
        self,
        in_channels: int,
        squeeze_ratio: float,
        expand_filters: int,
        pct_3x3: float,
        is_residual: bool = False,
    ):
        super().__init__()
        s_1x1 = int(squeeze_ratio * expand_filters)
        e_3x3 = int(expand_filters * pct_3x3)
        e_1x1 = expand_filters - e_3x3
        self.squeeze_1x1 = CNNBlock(in_channels, s_1x1, kernel_size=1, use_batch_norm=True)
        self.expand_1x1 = CNNBlock(s_1x1, e_1x1, kernel_size=1, use_batch_norm=True)
        self.expand_3x3 = CNNBlock(s_1x1, e_3x3, kernel_size=3, padding=1, use_batch_norm=True)
        self.is_residual = is_residual

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        squeeze_out = self.squeeze_1x1(x)
        expand_1x1_out = self.expand_1x1(squeeze_out)
        expand_3x3_out = self.expand_3x3(squeeze_out)
        out = torch.concat([expand_1x1_out, expand_3x3_out], dim=1)  # concat over channels
        if self.is_residual:
            return x + out
        return out


class SqueezeNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        base_e: int = 128,
        incr_e: int = 128,
        pct_3x3: float = 0.5,
        freq: int = 2,
        SR: float = 0.125,
        simple_bypass: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.base_e = base_e
        self.incr_e = incr_e
        self.pct_3x3 = pct_3x3
        self.freq = freq
        self.SR = SR

        # architecture, fb - fire block
        out_channels = 96
        n_fire_blocks = 8
        fb_expand_filters = [base_e + (incr_e * (i // freq)) for i in range(n_fire_blocks)]
        fb_in_channels = [out_channels] + fb_expand_filters
        is_residual = [False] + [(i % freq == 1 and simple_bypass) for i in range(1, n_fire_blocks)]
        self.fb_in_channels = fb_in_channels
        self.out_channels = fb_expand_filters[-1]
        conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=7, stride=2)
        maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        fire2 = FireBlock(fb_in_channels[0], SR, fb_expand_filters[0], pct_3x3, is_residual[0])
        fire3 = FireBlock(fb_in_channels[1], SR, fb_expand_filters[1], pct_3x3, is_residual[1])
        fire4 = FireBlock(fb_in_channels[2], SR, fb_expand_filters[2], pct_3x3, is_residual[2])
        maxpool4 = nn.MaxPool2d(kernel_size=3, stride=2)
        fire5 = FireBlock(fb_in_channels[3], SR, fb_expand_filters[3], pct_3x3, is_residual[3])
        fire6 = FireBlock(fb_in_channels[4], SR, fb_expand_filters[4], pct_3x3, is_residual[4])
        fire7 = FireBlock(fb_in_channels[5], SR, fb_expand_filters[5], pct_3x3, is_residual[5])
        fire8 = FireBlock(fb_in_channels[6], SR, fb_expand_filters[6], pct_3x3, is_residual[6])
        maxpool8 = nn.MaxPool2d(kernel_size=3, stride=2)
        fire9 = FireBlock(fb_in_channels[7], SR, fb_expand_filters[7], pct_3x3, is_residual[7])
        dropout9 = nn.Dropout2d(p=0.5)
        layers = [
            ("conv1", conv1),
            ("maxpool1", maxpool1),
            ("fire2", fire2),
            ("fire3", fire3),
            ("fire4", fire4),
            ("maxpool4", maxpool4),
            ("fire5", fire5),
            ("fire6", fire6),
            ("fire7", fire7),
            ("fire8", fire8),
            ("maxpool8", maxpool8),
            ("fire9", fire9),
            ("dropout9", dropout9),
        ]
        self.net = nn.Sequential(OrderedDict(layers))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DeepCNN(nn.Module):
    """Deep Convolutional Neural Network (CNN) constructed of many CNN blocks and ended with Global Average Pooling."""

    def __init__(
        self,
        in_channels: int,
        out_channels: List[int],
        kernels: List[_size_2_t],
        pool_kernels: List[_size_2_t],
        pool_type: Literal["Max", "Avg"] = "Max",
        use_batch_norm: bool = True,
        dropout: float = 0,
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
            use_batch_norm (bool, optional): Whether to use BN in CNN blocks. Defaults to True.
            dropout (float, optional): Dropout probability used in CNN blocks. Defaults to 0.
            activation (str, optional): Type of activation function used in CNN blocks. Defaults to 0.
        """
        super().__init__()
        self.out_channels = out_channels
        self.kernels = kernels
        self.pool_kernels = pool_kernels
        self.pool_type = pool_type
        self.use_batch_norm = use_batch_norm
        self.dropout = dropout
        self.activation = activation
        n_blocks = len(out_channels)
        fixed_params = dict(
            pool_type=pool_type,
            use_batch_norm=use_batch_norm,
            dropout=dropout,
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


def load_net() -> nn.Module:
    # backbone = SqueezeNet()
    backbone = DeepCNN(
        in_channels=3,
        out_channels=[16, 32, 64, 128],
        kernels=[3, 3, 3, 3],
        pool_kernels=[2, 2, 2, 2],
    )

    return nn.Sequential(
        backbone,
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(1, -1),
        nn.Linear(backbone.out_channels, 17),
        nn.LogSoftmax(dim=1),
    )
