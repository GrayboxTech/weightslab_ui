import pdb
from typing import List, Set, Dict
import torch
import torch as th
import torch.nn as nn
import torch.optim as optim

from torch.nn import functional as F
from torchvision import transforms as T
from torchvision import datasets as ds

from weightslab.experiment import Experiment

from weightslab.model_with_ops import NetworkWithOps
from weightslab.model_with_ops import DepType
from weightslab.modules_with_ops import Conv2dWithNeuronOps
from weightslab.modules_with_ops import LinearWithNeuronOps
from weightslab.modules_with_ops import BatchNorm2dWithNeuronOps

from weightslab.tracking import TrackingMode
from weightslab.tracking import add_tracked_attrs_to_input_tensor


from board import Dash


class FashionCNN(NetworkWithOps):
    def __init__(self):
        super(FashionCNN, self).__init__()
        self.tracking_mode = TrackingMode.DISABLED

        self.layer1 = Conv2dWithNeuronOps(
            in_channels=1, out_channels=2, kernel_size=3, padding=1)
        self.bnorm1 = BatchNorm2dWithNeuronOps(2)
        self.mpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.layer2 = Conv2dWithNeuronOps(
            in_channels=2, out_channels=4, kernel_size=3)
        self.bnorm2 = BatchNorm2dWithNeuronOps(4)
        self.mpool2 = nn.MaxPool2d(2)

        self.fc1 = LinearWithNeuronOps(in_features=4*6*6, out_features=6)
        # self.bnorm3 = BatchNorm2dWithNeuronOps(600)
        # self.drop1 = nn.Dropout(0.25)
        self.fc2 = LinearWithNeuronOps(in_features=6, out_features=8)
        # self.bnorm4 = BatchNorm2dWithNeuronOps(120)
        self.fc3 = LinearWithNeuronOps(in_features=8, out_features=10)
        self.softmax = nn.Softmax(dim=1)

    def children(self):
        return [
            self.layer1, self.bnorm1, self.layer2, self.bnorm2,
            self.fc1, self.fc2, self.fc3
        ]

    def define_deps(self):
        self.register_dependencies([
            (self.layer1, self.bnorm1, DepType.SAME),
            (self.bnorm1, self.layer2, DepType.INCOMING),
            (self.layer2, self.bnorm2, DepType.SAME),
            (self.bnorm2, self.fc1, DepType.INCOMING),
            (self.fc1, self.fc2, DepType.INCOMING),
            (self.fc2, self.fc3, DepType.INCOMING),
        ])

        self.flatten_conv_id = self.bnorm2.get_module_id()

    def forward(self, x, intermediary: Dict[int, th.Tensor] | None = None):
        self.maybe_update_age(x)
        x = self.layer1(x, intermediary=intermediary)
        x = self.bnorm1(x)
        x = F.relu(x)
        # x = F.leaky_relu(x, negative_slope=0.01)
        x = self.mpool1(x)

        x = self.layer2(x, intermediary=intermediary)
        x = self.bnorm2(x)
        x = F.relu(x)
        # x = F.leaky_relu(x, negative_slope=0.01)
        x = self.mpool2(x)

        x = x.view(x.size(0), -1)
        x = self.fc1(x, intermediary=intermediary)
        x = F.relu(x)
        # x = F.leaky_relu(x, negative_slope=0.01)
        # x = self.bnorm3(x)
        # x = self.drop1(x)
        x = self.fc2(x, intermediary=intermediary)
        x = F.relu(x)
        # x = F.leaky_relu(x, negative_slope=0.01)
        # x = self.bnorm4(x)
        output = self.fc3(x, skip_register=True, intermediary=None)

        one_hot = F.one_hot(
            output.argmax(dim=1), num_classes=self.fc3.out_features)

        if hasattr(x, 'in_id_batch') and hasattr(x, 'label_batch'):
            add_tracked_attrs_to_input_tensor(
                one_hot, in_id_batch=input.in_id_batch,
                label_batch=input.label_batch)
        self.fc3.register(one_hot)
        output = self.softmax(output)
        return output


train_set = ds.MNIST(
    "./data", download=True, transform=T.Compose([T.ToTensor()]))
test_set = ds.MNIST(
    "./data", download=True, train=False, transform=T.Compose([T.ToTensor()]))  

# device = th.device("cuda:0")
device = th.device("cpu")


def get_exp():
    model = FashionCNN()
    model.define_deps()
    exp = Experiment(
        model=model, optimizer_class=optim.Adam,
        train_dataset=train_set,
        eval_dataset=test_set,
        device=device, learning_rate=1e-3, batch_size=100,
        name="v0",
        root_log_dir='mnist-dev-exp7',
        logger=Dash("mnist-dev-exp7"),
        skip_loading=False)

    def stateful_difference_monitor_callback():
        exp.display_stats()

    exp.register_train_loop_callback(stateful_difference_monitor_callback)

    return exp