import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
import torchvision.transforms as tt
from torchvision.datasets import ImageFolder
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score
from weightslab.experiment import Experiment
from weightslab.model_with_ops import NetworkWithOps, DepType
from weightslab.modules_with_ops import (
    Conv2dWithNeuronOps,
    LinearWithNeuronOps,
    BatchNorm2dWithNeuronOps
)
from weightslab.tracking import TrackingMode
from board import Dash

class TinyImageNetNet(NetworkWithOps, nn.Module):
    def __init__(self):
        super().__init__()
        self.tracking_mode = TrackingMode.DISABLED

        self.conv1 = Conv2dWithNeuronOps(3,   64, kernel_size=3, padding=1)
        self.bn1   = BatchNorm2dWithNeuronOps(64)

        self.conv2 = Conv2dWithNeuronOps(64, 128, kernel_size=3, padding=1)
        self.bn2   = BatchNorm2dWithNeuronOps(128)

        self.conv3 = Conv2dWithNeuronOps(128, 256, kernel_size=3, padding=1)
        self.bn3   = BatchNorm2dWithNeuronOps(256)

        self.pool = nn.MaxPool2d(2)                 # down-sample ×2
        self.gap  = nn.AdaptiveAvgPool2d(1)         # 4×4 → 1×1
        self.dropout = nn.Dropout(p=0.20)

        self.fc = LinearWithNeuronOps(256, 200)

    def children(self):
        return [
            self.conv1, self.bn1,
            self.conv2, self.bn2,
            self.conv3, self.bn3,
            self.fc
        ]

    def define_deps(self):
        self.register_dependencies([
            (self.conv1, self.bn1, DepType.SAME),
            (self.bn1,   self.conv2, DepType.INCOMING),
            (self.conv2, self.bn2, DepType.SAME),
            (self.bn2,   self.conv3, DepType.INCOMING),
            (self.conv3, self.bn3, DepType.SAME),
            (self.bn3,   self.fc,   DepType.INCOMING)
        ])
        # last BN layer is the flatten boundary for WeightsLab
        self.flatten_conv_id = self.bn3.get_module_id()

    def forward(self, x):
        self.maybe_update_age(x)

        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # 64 → 32
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # 32 → 16
        x = self.pool(F.relu(self.bn3(self.conv3(x))))  # 16 → 8
        x = self.pool(x)                                # 8 → 4
        x = self.gap(x)                                 # 4×4 → 1×1
        x = x.view(x.size(0), -1)                       # (B, 192)
        x = self.dropout(x)
        return self.fc(x)                               # logits (B, 200)
    
# Normalization
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

transform_train = tt.Compose([
    tt.RandomHorizontalFlip(),
    tt.ToTensor(),
    tt.Normalize(mean, std)
])

transform_test = tt.Compose([
    tt.ToTensor(),
    tt.Normalize(mean, std)
])

train_set = ImageFolder('./data/tiny-imagenet-200/train', transform=transform_train)
test_set = ImageFolder('./data/tiny-imagenet-200/val/classified', transform=transform_test)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def custom_top5_accuracy(output, labels):
    top5 = output.topk(5, dim=1).indices
    correct = top5.eq(labels.view(-1, 1)).sum().item()
    return correct / labels.size(0)

metrics = {
    "acc": MulticlassAccuracy(num_classes=200, average="micro").to(device),
    "f1": MulticlassF1Score(num_classes=200, average="macro").to(device),
    "top5": custom_top5_accuracy
}

def custom_mse_loss(outputs, targets):
    return ((outputs - targets) ** 2).mean()

def get_exp():
    model = TinyImageNetNet()
    model.define_deps()
    model.to(device)

    exp = Experiment(
        model=model,
        optimizer_class=optim.Adam,
        train_dataset=train_set,
        eval_dataset=test_set,
        device=device,
        learning_rate=1e-3,
        batch_size=64,
        criterion=nn.CrossEntropyLoss(reduction='none'),  
        metrics=metrics,
        training_steps_to_do=30000,
        name="tinyimagenet_exp",
        root_log_dir="tinyimagenet-exp-test1",
        logger=Dash("tinyimagenet-exp-test1"),
        skip_loading=False
    )

    return exp
