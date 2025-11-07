import os
import time
import datetime
from typing import Dict
import torch
import torch as th
import torch.nn as nn
import torch.optim as optim

from torch.nn import functional as F
from torchvision import transforms as T
from torchvision import datasets as ds
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score

from board import Dash

from weightslab.experiment.experiment import Experiment
from weightslab.tests.torch_models import ResNet18, FashionCNNSequential


def get_exp():
    # Output directory

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # timestamp = "000"  # str(time.time())
    output_dir = r"out/" #r"C:\Users\GuillaumePelluet\Documents\Codes\grayBox\outputs\test_mnist"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory is {os.path.join(output_dir, timestamp)}")

    # Datasets
    train_set = ds.MNIST(os.path.join(output_dir, "data"), download=True, transform=T.Compose([T.ToTensor()]))
    test_set = ds.MNIST(os.path.join(output_dir, "data"), download=True, train=False, transform=T.Compose([T.ToTensor()]))

    # Models
    device = 'cpu'  # th.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = FashionCNNSequential()
    model = ResNet18()
    model.to(device)

    # Metrics & Loss
    metrics = {
        "acc": MulticlassAccuracy(num_classes=10, average="micro").to(device),
        "f1": MulticlassF1Score(num_classes=10, average="macro").to(device),
    }
    criterion = nn.CrossEntropyLoss(reduction='none')

    # Experiment
    params = {
        "model": model,
        "input_shape": model.input_shape,
        "train_dataset": train_set,
        "eval_dataset": test_set,
        "device": device,

        "optimizer_class": optim.Adam,
        "learning_rate": 1e-2,
        "batch_size": 128,
        "criterion": criterion,
        "metrics": metrics,
        "training_steps_to_do": 25600,
        "tqdm_display": True,
        "name": "v0",
        "skip_loading": False,
        "root_log_dir": os.path.join(output_dir, timestamp),
        "logger": Dash(os.path.join(output_dir, timestamp))
    }
    exp = Experiment(**params)
    return exp


if __name__ == "__main__":
    print('hello world')

    # Get Experiment
    get_exp()
