import os
import datetime
import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import transforms as T
from torchvision import datasets as ds
from torchvision.models import resnet18, ResNet18_Weights

from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score

from .board import Dash
from weightslab.experiment.experiment import Experiment


class TinyResNet18(nn.Module):
    """
    Very small ResNet18-derived model:

    stem:   conv1 -> bn1 -> relu -> maxpool
    body:   ONLY the first BasicBlock from layer1
    head:   global avgpool -> fc(64 -> num_classes)

    Still allows loading ResNet18 pretrained weights.
    """

    # for CIFAR-10
    input_shape = (1, 3, 32, 32)

    def __init__(
        self,
        pretrained: bool = True,
        in_channels: int = 3,
        num_classes: int = 10,
    ):
        super().__init__()

        # ---- 1) Get base ResNet18 ----
        if pretrained:
            base = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        else:
            base = resnet18(weights=None)

        # ---- 2) Stem ----
        # Adapt conv1 to arbitrary in_channels
        if in_channels != 3:
            old_weight = base.conv1.weight.data  # [64, 3, 7, 7]
            self.conv1 = nn.Conv2d(
                in_channels,
                64,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            )
            with torch.no_grad():
                if in_channels == 1:
                    # average RGB channels -> single channel
                    w = old_weight.mean(dim=1, keepdim=True)  # [64,1,7,7]
                    self.conv1.weight[:] = w
                else:
                    # repeat / truncate channels
                    repeat = (in_channels + 2) // 3
                    w = old_weight.repeat(1, repeat, 1, 1)[:, :in_channels]
                    self.conv1.weight[:] = w
        else:
            self.conv1 = base.conv1  # direct reuse

        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool

        # ---- 3) Body: only FIRST block of layer1 ----
        self.block1 = base.layer1[0]  # BasicBlock(64 -> 64)

        # ---- 4) Tiny head ----
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)  # 64 channels after block1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.block1(x)  # only one residual block

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


# ---- Experiment definition ----

def get_exp():
    # 1) Output + run directory
    output_dir = "out"
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(output_dir, timestamp)
    data_dir = os.path.join(output_dir, "data")

    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    print(f"Output directory is {run_dir}")

    # 2) Dataset (CIFAR-10)
    # kept transforms simple to mirror your MNIST setup
    transform = T.ToTensor()
    train_set = ds.CIFAR10(data_dir, train=True, download=True, transform=transform)
    test_set = ds.CIFAR10(data_dir, train=False, download=True, transform=transform)

    # 3) Device
    device = "cpu"  # or: 'cuda' if torch.cuda.is_available() else 'cpu'

    # 4) Create the tiny ResNet18 model (plain PyTorch)
    model = TinyResNet18(
        pretrained=False,
        in_channels=3,
        num_classes=10,
    ).to(device)

    print(">>> Raw model before WatcherEditor:")
    print(model)

    # 5) Metrics & Loss
    metrics = {
        "acc": MulticlassAccuracy(num_classes=10, average="micro").to(device),
        "f1": MulticlassF1Score(num_classes=10, average="macro").to(device),
    }
    criterion = nn.CrossEntropyLoss(reduction="none")

    # 6) Experiment params
    params = {
        "model": model,
        "input_shape": model.input_shape,
        "train_dataset": train_set,
        "eval_dataset": test_set,
        "device": device,

        "optimizer_class": optim.Adam,
        "learning_rate": 1e-3,
        "batch_size": 128,
        "criterion": criterion,
        "metrics": metrics,
        "training_steps_to_do": 25600,
        "tqdm_display": True,
        "name": "resnet18_small_v0",
        "skip_loading": False,
        "root_log_dir": run_dir,
        "logger": Dash(run_dir),
    }

    return Experiment(**params)


if __name__ == "__main__":
    exp = get_exp()
