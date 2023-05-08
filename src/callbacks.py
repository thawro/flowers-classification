import torch
from torchmetrics import ConfusionMatrix
from typing import List
import wandb
import pytorch_lightning as pl
from data import img_unnormalizer
from plotly.subplots import make_subplots
import plotly.express as px
import random
import plotly.figure_factory as ff


def plot_multiclass_confusion_matrix(targets: torch.Tensor, probs: torch.Tensor, labels: List[str]):
    num_classes = len(labels)
    confmat = ConfusionMatrix(task="multiclass", num_classes=num_classes, normalize="pred")
    cm = confmat(preds=probs, target=targets).numpy()[::-1]
    fig = ff.create_annotated_heatmap(
        cm.tolist(),
        x=labels,
        y=labels[::-1],
        annotation_text=cm.round(2),
        colorscale="Blues",
        showscale=False,
    )
    size = 60 * num_classes
    fig.update_layout(
        width=size,
        height=size,
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def plot_imgs_probs_plotly(
    images: torch.Tensor,
    targets: torch.Tensor,
    probs: torch.Tensor,
    labels: List[str],
    n_best: int = 5,
):
    if images.shape[-1] == 1:  # GREYSCALE
        images = images.repeat(1, 1, 1, 3)
    palette = px.colors.qualitative.Plotly
    n_examples = len(images)
    fig = make_subplots(
        rows=2, cols=n_examples, vertical_spacing=0.05, horizontal_spacing=0.05, row_heights=[1, 2]
    )
    for col, (img, target, prob) in enumerate(zip(images, targets, probs)):
        pred = int(prob.argmax().item())
        colors = [palette[0]] * len(labels)
        if pred == target:
            colors[pred] = palette[2]
        else:
            colors[pred] = palette[1]
            colors[target] = palette[9]
        sorted_probs = sorted(prob, reverse=False)[-n_best:]
        sorted_labels = [label for _, label in sorted(zip(prob, labels), reverse=False)][-n_best:]
        sorted_colors = [color for _, color in sorted(zip(prob, colors), reverse=False)][-n_best:]

        fig.add_bar(
            x=sorted_probs,
            y=sorted_labels,
            orientation="h",
            marker_color=sorted_colors,
            row=1,
            col=col + 1,
            text=sorted_labels,
            textposition="inside",
            insidetextanchor="start",
            insidetextfont=dict(family="Arial", size=14, color="black"),
            outsidetextfont=dict(family="Arial", size=14, color="black"),
        )
        fig.add_image(z=img, zmin=[0] * 4, zmax=[1] * 4, row=2, col=col + 1, name=labels[target])
        fig.update_layout(showlegend=False, margin=dict(l=0, r=0, t=0, b=0))
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)
    return fig


class ConfusionMatrixLogger(pl.Callback):
    def __init__(
        self,
        classes: List[str],
    ):
        super().__init__()
        self.classes = classes

    def on_test_epoch_end(self, trainer, pl_module):
        outputs = pl_module.outputs["test"]
        probs = torch.concat([output["probs"] for output in outputs]).cpu()
        targets = torch.concat([output["targets"] for output in outputs]).cpu()
        fig = plot_multiclass_confusion_matrix(targets=targets, probs=probs, labels=self.classes)
        pl_module.logged_metrics["test/confusion_matrix"] = wandb.Plotly(fig)


class ExamplePredictionsLogger(pl.Callback):
    def __init__(
        self,
        classes: List[str],
        num_examples: int = 8,
    ):
        super().__init__()
        self.num_examples = num_examples
        self.classes = classes

    def on_test_epoch_end(self, trainer, pl_module):
        examples = pl_module.examples["test"]
        probs, targets = examples["probs"], examples["targets"]
        images = examples["images"]
        images = img_unnormalizer(images).permute(0, 2, 3, 1)

        idxs = random.choices(range(len(targets)), k=self.num_examples)
        fig = plot_imgs_probs_plotly(
            images=images[idxs],
            targets=targets[idxs],
            probs=probs[idxs],
            labels=self.classes,
        )
        pl_module.logged_metrics[f"test/random_examples"] = wandb.Plotly(fig)
