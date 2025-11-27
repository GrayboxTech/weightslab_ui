import os
import datetime
import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import transforms as T
from torchvision import datasets as ds
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score

from .board import Dash
from weightslab.experiment.experiment import Experiment
from weightslab.weightslab.baseline_models.pytorch.models import ResNet18, FashionCNNSequential

# MNIST EXPERIMENT

def get_exp():
    # 1) Output + run directory
    output_dir = "out"
    # set to a fixed string (e.g. "dev") if you want to reuse the same folder
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(output_dir, timestamp)
    data_dir = os.path.join(output_dir, "data")

    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    print(f"Output directory is {run_dir}")

    # 2) Datasets 
    transform = T.ToTensor()
    train_set = ds.MNIST(data_dir, download=True, transform=transform)
    test_set  = ds.MNIST(data_dir, download=True, train=False, transform=transform)

    # 3) Device
    device = 'cpu'  # change to: 'cuda' if torch.cuda.is_available() else 'cpu'

    # 4) Model
    model = FashionCNNSequential()
    model.to(device)

    # 5) Metrics & Loss
    metrics = {
        "acc": MulticlassAccuracy(num_classes=10, average="micro").to(device),
        "f1":  MulticlassF1Score(num_classes=10, average="macro").to(device),
    }
    criterion = nn.CrossEntropyLoss(reduction='none')

    # 6) Experiment params
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
        "root_log_dir": run_dir,
        "logger": Dash(run_dir),
    }
    return Experiment(**params)


if __name__ == "__main__":
    get_exp()
