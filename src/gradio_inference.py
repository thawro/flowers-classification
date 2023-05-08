import gradio as gr
import torch
import torchvision.transforms as T
from .data import MEAN_IMAGENET, STD_IMAGENET, LABELS


examples = [
    f"data/examples/{example}.jpg"
    for example in ["crocus", "dandelion", "snowdrop", "sunflower", "pansy", "fritillary"]
]

transform = T.Compose(
    [
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(MEAN_IMAGENET, STD_IMAGENET),
    ]
)
model = torch.jit.load("models/DeepCNN.pt")
model.eval()


def predict(inp):
    inp = transform(inp).unsqueeze(0)
    log_probs = model(inp)[0]
    probs = torch.exp(log_probs)
    confidences = {LABELS[i]: float(probs[i]) for i in range(len(LABELS))}
    return confidences


gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=5),
    examples=examples,
).launch()
