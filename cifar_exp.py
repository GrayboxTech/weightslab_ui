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


from board import Dash


        #         # print("Model.forward", self.seen_samples)
        #         self.maybe_update_age(xb)
        #         out = self.conv1(xb)
        #         out = self.conv2(out)
        #         out = self.res1(out)# + out
        #         out = self.conv3(out)
        #         out = self.conv4(out)
        #         out = self.res2(out)# + out
        #         out = self.conv5(out)
        #         out = self.res3(out)# + out
        #         out = self.classifier(out)
        #         return out


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
        self.fc = LinearWithNeuronOps(4 * 4 * 4, 10)

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





train_data = ds.CIFAR10('./', train=True, download=True)

# Stick all the images together to form a 1600000 X 32 X 3 array
x = np.concatenate([np.asarray(train_data[i][0]) for i in range(len(train_data))])

# calculate the mean and std along the (0, 1) axes
mean = np.mean(x, axis=(0, 1))/255
std = np.std(x, axis=(0, 1))/255
# the the mean and std
mean=mean.tolist()
std=std.tolist()


mean=[0.4914, 0.4822, 0.4465]
std=[0.2470, 0.2435, 0.2616]

transform_train = tt.Compose(
    [
        # tt.RandomCrop(32, padding=4, padding_mode='reflect'),
        tt.RandomHorizontalFlip(),
        tt.ToTensor(), 
        tt.Normalize(mean, std, inplace=True)
    ]
)
transform_test = tt.Compose([
    tt.ToTensor(),
    tt.Normalize(mean, std)
])


train_set = ds.CIFAR10(
    "./data", download=True, transform=transform_train)
test_set = ds.CIFAR10(
    "./data", download=True, train=False, transform=transform_test)

device = th.device("cuda:0")

def get_exp():
    model = SmallCIFARNet()
    model.define_deps()
    model.to(device)
    exp = Experiment(
        model=model, optimizer_class=optim.Adam,
        train_dataset=train_set,
        eval_dataset=test_set,
        device=device, learning_rate=1e-4, batch_size=2048,
        training_steps_to_do=200000,
        name="v0",
        root_log_dir='cifar10',
        logger=Dash("cifar10"),
        skip_loading=False)

    def stateful_difference_monitor_callback():
        exp.display_stats()

    return exp


# exp = get_exp()
# import pdb; pdb.set_trace()
# exp.model.add_neurons(layer_id=0, neuron_count=4)
# exp.model.add_neurons(layer_id=2, neuron_count=4)
# exp.model.add_neurons(layer_id=4, neuron_count=4)
