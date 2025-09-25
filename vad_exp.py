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
from torchmetrics.classification import BinaryAccuracy, BinaryF1Score


from board import Dash


class ConvNet(NetworkWithOps, nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.tracking_mode = TrackingMode.DISABLED

        N = 8
        # Convolution block 1
        self.conv1 = Conv2dWithNeuronOps(in_channels=3, out_channels=N, kernel_size=3, padding=1)
        self.bnorm1 = BatchNorm2dWithNeuronOps(N)

        # Convolution block 2
        self.conv2 = Conv2dWithNeuronOps(in_channels=N, out_channels=N, kernel_size=21, padding=11, stride=11)
        self.bnorm2 = BatchNorm2dWithNeuronOps(N)

        # Convolution block 3
        self.conv3 = Conv2dWithNeuronOps(in_channels=N, out_channels=N, kernel_size=21, padding=11, stride=11)
        self.bnorm3 = BatchNorm2dWithNeuronOps(N)

        # Convolution block 4
        # self.conv4 = Conv2dWithNeuronOps(in_channels=N, out_channels=N, kernel_size=3, padding=1)
        # self.bnorm4 = BatchNorm2dWithNeuronOps(N)

        # # Convolution block 5
        # self.conv5 = Conv2dWithNeuronOps(in_channels=N, out_channels=N, kernel_size=3, padding=1)
        # self.bnorm5 = BatchNorm2dWithNeuronOps(N)

        # # Convolution block 6
        # self.conv6 = Conv2dWithNeuronOps(in_channels=N, out_channels=N, kernel_size=3, padding=1)
        # self.bnorm6 = BatchNorm2dWithNeuronOps(N)

        # # Convolution block 7
        # self.conv7 = Conv2dWithNeuronOps(in_channels=N, out_channels=N, kernel_size=3, padding=1)
        # self.bnorm7 = BatchNorm2dWithNeuronOps(N)

        # Fully connected output
        self.fc = LinearWithNeuronOps(N * 3 * 3, 1)

    def children(self):
        return [
            self.conv1, self.bnorm1, self.conv2, self.bnorm2,
            self.conv3, self.bnorm3,

            # self.conv4, self.bnorm4,
            # self.conv5, self.bnorm5, self.conv6, self.bnorm6,
            # self.conv7, self.bnorm7,

            self.fc
        ]

    def define_deps(self):
        self.register_dependencies([
            (self.conv1, self.bnorm1, DepType.SAME),
            (self.bnorm1, self.conv2, DepType.INCOMING),
            (self.conv2, self.bnorm2, DepType.SAME),
            (self.bnorm2, self.conv3, DepType.INCOMING),
            (self.conv3, self.bnorm3, DepType.SAME),
            # (self.bnorm3, self.conv4, DepType.INCOMING),
            # (self.conv4, self.bnorm4, DepType.SAME),
            # (self.bnorm4, self.conv5, DepType.INCOMING),
            # (self.conv5, self.bnorm5, DepType.SAME),
            # (self.bnorm5, self.conv6, DepType.INCOMING),
            # (self.conv6, self.bnorm6, DepType.SAME),
            # (self.bnorm6, self.conv7, DepType.INCOMING),
            # (self.conv7, self.bnorm7, DepType.SAME),
            # (self.bnorm7, self.fc, DepType.INCOMING),
            (self.bnorm3, self.fc, DepType.INCOMING),
        ])

        # self.flatten_conv_id = self.bnorm7.get_module_id()
        self.flatten_conv_id = self.bnorm3.get_module_id()

    def forward(self, x, intermediary_outputs=None):
        self.maybe_update_age(x)
        # Block 1: conv -> BN -> ReLU -> max pool
        # x = F.relu(self.bnorm1(self.conv1(x, intermediary=intermediary_outputs)))
        # x = F.max_pool2d(x, 2)  # reduces spatial size from 256 -> 128

        x = self.conv1(x, intermediary=intermediary_outputs)
        x = self.bnorm1(x, intermediary=intermediary_outputs)   
        x = F.relu(x)
        x = F.max_pool2d(x, 2)

        # Block 2
        #  x = F.relu(self.bnorm2(self.conv2(x, intermediary=intermediary_outputs)))
        # x = F.max_pool2d(x, 7)  # 128 -> 64

        x = self.conv2(x, intermediary=intermediary_outputs)
        x = self.bnorm2(x, intermediary=intermediary_outputs)  
        x = F.relu(x)

        # Block 3
        # x = F.relu(self.bnorm3(self.conv3(x, intermediary=intermediary_outputs)))
        # x = F.max_pool2d(x, 7)  # 64 -> 32

        x = self.conv3(x, intermediary=intermediary_outputs)
        x = self.bnorm3(x, intermediary=intermediary_outputs)  
        x = F.relu(x)

        # # Block 4
        # x = F.relu(self.bnorm4(self.conv4(x, intermediary=intermediary_outputs)))
        # x = F.max_pool2d(x, 2)  # 32 -> 16

        # # Block 5
        # x = F.relu(self.bnorm5(self.conv5(x, intermediary=intermediary_outputs)))
        # x = F.max_pool2d(x, 2)  # 16 -> 8

        # # Block 6
        # x = F.relu(self.bnorm6(self.conv6(x, intermediary=intermediary_outputs)))
        # x = F.max_pool2d(x, 2)  # 8 -> 4

        # # Block 7
        # x = F.relu(self.bnorm7(self.conv7(x, intermediary=intermediary_outputs)))
        # x = F.max_pool2d(x, 2)  # 4 -> 2

        # Flatten
        x = x.view(x.size(0), -1)  # shape = [batch_size, 4*4*4]
        out = self.fc(x)
        return out.squeeze(1)

IM_MEAN = (.5533, .5829, .5946)
IM_STD  = (.1527, .1628, .1726)

train_transform = T.Compose([
    T.ToTensor(),
    T.Normalize(IM_MEAN, IM_STD),
])

val_transform = T.Compose([
    T.ToTensor(),
    T.Normalize(IM_MEAN, IM_STD),
])

root_dir = 'VAD'


train_dataset = ds.ImageFolder(
    os.path.join(root_dir, "train"), transform=train_transform)
val_dataset = ds.ImageFolder(
    os.path.join(root_dir, "test"), transform=val_transform)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


metrics = {
    "acc": BinaryAccuracy().to(device),
    # "f1": BinaryF1Score().to(device),
}

def get_exp():
    model = ConvNet()
    model.define_deps()
    model.to(device)
    exp = Experiment(
        model=model, optimizer_class=optim.Adam,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        device=device, learning_rate=2e-4, batch_size=64,
        training_steps_to_do=200000,
        name="v0",
        metrics=metrics,
        root_log_dir='vad',
        logger=Dash("vad"),
        criterion=nn.BCEWithLogitsLoss(
            reduction='none',
            # pos_weight=th.tensor([0.5,], device=device)
        ),
        skip_loading=False)

    def stateful_difference_monitor_callback():
        exp.display_stats()

    return exp


if __name__ == "__main__":
    import pdb; pdb.set_trace()
    exp = get_exp()
    exp.train_step_or_eval_full()