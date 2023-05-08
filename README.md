# flowers-classification
Flower species classification using simple Deep Convolutional Neural Network (DeepCNN). DeepCNN is trained with [17 Category Flower Dataset](https://www.robots.ox.ac.uk/~vgg/data/flowers/17/index.html). All experiments are logged to the WandB project which can be found [here](https://wandb.ai/thawro/flowers-classification?workspace=user-thawro)

## Tech stack
* [PyTorch](https://pytorch.org/) - neural networks architectures and datasets classes
* [PyTorch Lightning](https://www.pytorchlightning.ai/index.html) - model training and evaluation
* [plotly](https://plotly.com/) - visualizations
* [WandB](https://docs.wandb.ai/) - metrics, visualizations and model logging
* [torchmetrics](https://torchmetrics.readthedocs.io/en/stable/) - metrics calculation
* [gradio](https://gradio.app/) - application used to show how model works in real world

## Model training 
### Command
```bat
make train_model
```
### Screenshots from wandb
train vs val metrics (loss, accuracy and fscore)
![alt text](https://github.com/thawro/flowers-classification/blob/main/plots/train_val_loss.png
![alt text](https://github.com/thawro/flowers-classification/blob/main/plots/train_val_accuracy.png
![alt text](https://github.com/thawro/flowers-classification/blob/main/plots/train_val_fscore.png

confusion matrix (test set)
![alt text](https://github.com/thawro/flowers-classification/blob/main/plots/test_confusion_matrix.png

example predictions (test set)
![alt text](https://github.com/thawro/flowers-classification/blob/main/plots/example_test_preds.png

## Gradio inference 
### Command
```bat
make gradio_inference
```
### Screenshots from gradio
![alt text](https://github.com/thawro/flowers-classification/blob/main/plots/gradio_dandelion.png
![alt text](https://github.com/thawro/flowers-classification/blob/main/plots/gradio_sunflower.png
![alt text](https://github.com/thawro/flowers-classification/blob/main/plots/gradio_crocus.png
