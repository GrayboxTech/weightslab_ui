import os
from typing import List, Set, Dict
import torch
import torch as th
import torch.nn as nn
import torch.optim as optim

import numpy as np

from torch.nn import functional as F
from torchvision import transforms as T
from torchvision import datasets as ds
import torchvision.transforms as tt

from weightslab.experiment import Experiment
from weightslab.model_with_ops import NetworkWithOps
from weightslab.model_with_ops import DepType
from weightslab.modules_with_ops import Conv2dWithNeuronOps
from weightslab.modules_with_ops import LinearWithNeuronOps
from weightslab.modules_with_ops import BatchNorm2dWithNeuronOps
from weightslab.modules_with_ops import LayerWiseOperations

from weightslab.tracking import TrackingMode
from weightslab.tracking import add_tracked_attrs_to_input_tensor

from torch.utils.data import DataLoader, random_split

from board import Dash


class SmallCIFARNet(NetworkWithOps, nn.Module):
    def __init__(self):
        super(SmallCIFARNet, self).__init__()
        self.tracking_mode = TrackingMode.DISABLED
        # Convolution block 1
        self.conv1 = Conv2dWithNeuronOps(in_channels=3, out_channels=4, kernel_size=3, padding=1)
        self.bnorm1 = BatchNorm2dWithNeuronOps(4)

        # Convolution block 2
        self.conv2 = Conv2dWithNeuronOps(in_channels=4, out_channels=4, kernel_size=3, padding=1)
        self.bnorm2 = BatchNorm2dWithNeuronOps(4)

        # Convolution block 3
        self.conv3 = Conv2dWithNeuronOps(in_channels=4, out_channels=4, kernel_size=3, padding=1)
        self.bnorm3 = BatchNorm2dWithNeuronOps(4)

        # Fully connected output
        # After three max-pool layers on 32x32 -> 4x4 feature maps
        self.fc = LinearWithNeuronOps(4096, 2)

    def children(self):
        return [
            self.conv1, self.bnorm1, self.conv2, self.bnorm2,
            self.conv3, self.bnorm3, self.fc
        ]
        
    def define_deps(self):
        self.register_dependencies([
            (self.conv1, self.bnorm1, DepType.SAME),
            (self.bnorm1, self.conv2, DepType.INCOMING),
            (self.conv2, self.bnorm2, DepType.SAME),
            (self.bnorm2, self.conv3, DepType.INCOMING),
            (self.conv3, self.bnorm3, DepType.SAME),
            (self.bnorm3, self.fc, DepType.INCOMING),
        ])

        self.flatten_conv_id = self.bnorm3.get_module_id()


    def forward(self, x):
        self.maybe_update_age(x)
        # Block 1: conv -> BN -> ReLU -> max pool
        x = F.relu(self.bnorm1(self.conv1(x)))
        x = F.max_pool2d(x, 2)  # reduces spatial size from 32 -> 16

        # Block 2
        x = F.relu(self.bnorm2(self.conv2(x)))
        x = F.max_pool2d(x, 2)  # 16 -> 8

        # Block 3
        x = F.relu(self.bnorm3(self.conv3(x)))
        x = F.max_pool2d(x, 2)  # 8 -> 4

        # Flatten
        x = x.view(x.size(0), -1)  # shape = [batch_size, 128*4*4]
        out = self.fc(x)
        return out


transform = T.Compose([T.Resize((256, 256)), T.ToTensor(),])

# Load dataset
full_dataset = ds.ImageFolder(
    root="/home/rotaru/Desktop/GRAYBOX/sales/pitch/prepare/cad_models_dataset/",
    # "cad_models_dataset/",
    transform=transform)

# Calculate sizes for train and validation sets
total_size = len(full_dataset)
train_size = int(0.8 * total_size)
val_size = total_size - train_size

# Split the dataset
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
train_dataset = train_dataset.dataset
val_dataset = val_dataset.dataset

# import pdb; pdb.set_trace()


device = th.device("cuda:0")

def get_exp():
    model = SmallCIFARNet()
    model.define_deps()
    model.to(device)
    exp = Experiment(
        model=model, optimizer_class=optim.Adam,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        device=device, learning_rate=1e-4, batch_size=2048,
        training_steps_to_do=200000,
        name="v0",
        root_log_dir='cad_models',
        logger=Dash("cad_models"),
        skip_loading=False)

    def stateful_difference_monitor_callback():
        exp.display_stats()

    return exp
