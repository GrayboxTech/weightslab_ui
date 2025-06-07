import os
from typing import List, Dict
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as tt
from torchvision.datasets import ImageFolder
from torch.nn import functional as F

from weightslab.experiment import Experiment
from weightslab.model_with_ops import NetworkWithOps, DepType
from weightslab.modules_with_ops import Conv2dWithNeuronOps, LinearWithNeuronOps, BatchNorm2dWithNeuronOps
from weightslab.tracking import TrackingMode
from board import Dash

class TinyImageNetNet(NetworkWithOps, nn.Module):
    def __init__(self):
        super().__init__()
        self.tracking_mode = TrackingMode.DISABLED
        self.conv1 = Conv2dWithNeuronOps(3, 16, kernel_size=3, padding=1)
        self.bnorm1 = BatchNorm2dWithNeuronOps(16)
        self.conv2 = Conv2dWithNeuronOps(16, 32, kernel_size=3, padding=1)
        self.bnorm2 = BatchNorm2dWithNeuronOps(32)
        self.fc = LinearWithNeuronOps(32 * 8 * 8, 200)  # TinyImageNet has 200 classes

    def children(self):
        return [self.conv1, self.bnorm1, self.conv2, self.bnorm2, self.fc]

    def define_deps(self):
        self.register_dependencies([
            (self.conv1, self.bnorm1, DepType.SAME),
            (self.bnorm1, self.conv2, DepType.INCOMING),
            (self.conv2, self.bnorm2, DepType.SAME),
            (self.bnorm2, self.fc, DepType.INCOMING),
        ])
        self.flatten_conv_id = self.bnorm2.get_module_id()

    def forward(self, x):
        self.maybe_update_age(x)
        x = F.relu(self.bnorm1(self.conv1(x)))
        x = F.max_pool2d(x, 2)  # 64x64 -> 32x32
        x = F.relu(self.bnorm2(self.conv2(x)))
        x = F.max_pool2d(x, 4)  # 32x32 -> 8x8
        x = x.view(x.size(0), -1)
        return self.fc(x)

# TinyImageNet mean/std (approx ImageNet)
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
val_dir = './data/tiny-imagenet-200/val/images'

# TinyImageNet val folder is messy; you may need to reorganize val images by class.
# Use train_set for both if needed for a quick test:
test_set = ImageFolder('./data/tiny-imagenet-200/train', transform=transform_test)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        batch_size=32,
        training_steps_to_do=25600,
        name="tinyimagenet_exp",
        root_log_dir="tinyimagenet-exp1",
        logger=Dash("tinyimagenet-exp1"),
        skip_loading=False
    )

    return exp
