import os
import datetime
import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import transforms as T
from torchvision import datasets as ds
from torch.nn import functional as F
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score

from weightslab.experiment.experiment import Experiment
from .board import Dash


class ConvNet(nn.Module):
    def __init__(self, num_classes: int = 78):
        super().__init__()

        # Convolution block 1
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=3, padding=1)
        self.bnorm1 = nn.BatchNorm2d(12)

        # Convolution block 2
        self.conv2 = nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, padding=1)
        self.bnorm2 = nn.BatchNorm2d(12)

        # Convolution block 3
        self.conv3 = nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, padding=1)
        self.bnorm3 = nn.BatchNorm2d(12)

        # Convolution block 4
        self.conv4 = nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, padding=1)
        self.bnorm4 = nn.BatchNorm2d(12)

        # Convolution block 5
        self.conv5 = nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, padding=1)
        self.bnorm5 = nn.BatchNorm2d(12)

        # Convolution block 6
        self.conv6 = nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, padding=1)
        self.bnorm6 = nn.BatchNorm2d(12)

        # Convolution block 7
        self.conv7 = nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, padding=1)
        self.bnorm7 = nn.BatchNorm2d(12)

        # Fully connected output (12 channels * 2 * 2 = 48)
        self.fc = nn.Linear(48, num_classes)

        # IMPORTANT: needed so backend / WatcherEditor can create dummy input
        # Adjust spatial size if your images are not 256x256
        self.input_shape = (1, 3, 256, 256)

    def forward(self, x):
        # Block 1: conv -> BN -> ReLU -> max pool
        x = F.relu(self.bnorm1(self.conv1(x)))
        x = F.max_pool2d(x, 2)  # 256 -> 128

        # Block 2
        x = F.relu(self.bnorm2(self.conv2(x)))
        x = F.max_pool2d(x, 2)  # 128 -> 64

        # Block 3
        x = F.relu(self.bnorm3(self.conv3(x)))
        x = F.max_pool2d(x, 2)  # 64 -> 32

        # Block 4
        x = F.relu(self.bnorm4(self.conv4(x)))
        x = F.max_pool2d(x, 2)  # 32 -> 16

        # Block 5
        x = F.relu(self.bnorm5(self.conv5(x)))
        x = F.max_pool2d(x, 2)  # 16 -> 8

        # Block 6
        x = F.relu(self.bnorm6(self.conv6(x)))
        x = F.max_pool2d(x, 2)  # 8 -> 4

        # Block 7
        x = F.relu(self.bnorm7(self.conv7(x)))
        x = F.max_pool2d(x, 2)  # 4 -> 2

        # Flatten: [B, 12, 2, 2] -> [B, 48]
        x = x.view(x.size(0), -1)
        out = self.fc(x)
        return out


IM_MEAN = (0.6312528, 0.4949005, 0.3298562)
IM_STD  = (0.0721354, 0.0712461, 0.0598827)

train_transform = T.Compose([
    T.ToTensor(),
    T.Normalize(IM_MEAN, IM_STD),
])

val_transform = T.Compose([
    T.ToTensor(),
    T.Normalize(IM_MEAN, IM_STD),
])


def get_datasets(data_dir: str):
    """
    Expecting the structure:
        data_dir/
          ycb_datasets/
            train/
            val/
    """
    root_dir = os.path.join(data_dir, "ycb_datasets")

    train_dataset = ds.ImageFolder(
        os.path.join(root_dir, "train"),
        transform=train_transform
    )
    val_dataset = ds.ImageFolder(
        os.path.join(root_dir, "val"),
        transform=val_transform
    )
    return train_dataset, val_dataset


def get_exp():
    # 1) Output + run directory
    output_dir = "out"
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(output_dir, timestamp)
    data_dir = os.path.join(output_dir, "data")

    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    print(f"Output directory is {run_dir}")
    print(f"Data directory is {data_dir}")

    # 2) Datasets
    train_dataset, val_dataset = get_datasets(data_dir)

    # 3) Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 4) Model
    model = ConvNet(num_classes=78)
    model.to(device)

    # 5) Metrics & Loss
    metrics: Dict[str, nn.Module] = {
        "acc": MulticlassAccuracy(num_classes=78, average="micro").to(device),
        # "f1": MulticlassF1Score(num_classes=78, average="macro").to(device),
    }
    criterion = nn.CrossEntropyLoss(reduction="none")

    # 6) Experiment params (mirrors your MNIST experiment)
    params = {
        "model": model,
        "input_shape": model.input_shape,
        "train_dataset": train_dataset,
        "eval_dataset": val_dataset,
        "device": device,

        "optimizer_class": optim.Adam,
        "learning_rate": 1e-3,
        "batch_size": 256,
        "criterion": criterion,
        "metrics": metrics,
        "training_steps_to_do": 200_000,
        "tqdm_display": True,
        "name": "v0",
        "skip_loading": False,
        "root_log_dir": run_dir,
        "logger": Dash(run_dir),
    }

    return Experiment(**params)



if __name__ == "__main__":
    exp = get_exp()
    exp.train_step_or_eval_full()
