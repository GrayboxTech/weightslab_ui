import os
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


class NanoModel(NetworkWithOps):
    def __init__(self):
        super(NanoModel, self).__init__()
        self.tracking_mode = TrackingMode.DISABLED
        N = 4

        self.layer1 = Conv2dWithNeuronOps(
            in_channels=3, out_channels=N, kernel_size=3, padding=1)
        self.bnorm1 = BatchNorm2dWithNeuronOps(N)
        self.mpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.layer2 = Conv2dWithNeuronOps(
            in_channels=N, out_channels=N, kernel_size=3)
        self.bnorm2 = BatchNorm2dWithNeuronOps(N)
        self.mpool2 = nn.MaxPool2d(2)

        self.layer3 = Conv2dWithNeuronOps(
            in_channels=N, out_channels=N, kernel_size=3)
        self.bnorm3 = BatchNorm2dWithNeuronOps(N)
        self.mpool3 = nn.MaxPool2d(2)


        self.layer34 = Conv2dWithNeuronOps(
            in_channels=N, out_channels=N, kernel_size=3)
        self.bnorm34 = BatchNorm2dWithNeuronOps(N)
        self.mpool34 = nn.MaxPool2d(2)


        self.layer4 = Conv2dWithNeuronOps(
            in_channels=N, out_channels=N, kernel_size=3)
        self.bnorm4 = BatchNorm2dWithNeuronOps(N)
        self.mpool4 = nn.MaxPool2d(2)

        self.layer5 = Conv2dWithNeuronOps(
            in_channels=N, out_channels=N, kernel_size=3)
        self.bnorm5 = BatchNorm2dWithNeuronOps(N)
        self.mpool5 = nn.MaxPool2d(2)

        self.fc1 = LinearWithNeuronOps(in_features=N*5*5, out_features=N)
        # self.bnorm3 = BatchNorm2dWithNeuronOps(10)
        self.drop1 = nn.Dropout(0.25)
        self.fc2 = LinearWithNeuronOps(in_features=N, out_features=N)
        # self.bnorm4 = BatchNorm2dWithNeuronOps(120)
        self.fc3 = LinearWithNeuronOps(in_features=N, out_features=200)
        self.softmax = nn.Softmax(dim=1)

    def children(self):
        return [
            self.layer1, self.bnorm1, self.layer2, self.bnorm2, self.layer3,
            self.bnorm3, self.layer34, self.bnorm34, self.layer4, self.bnorm4, self.layer5, self.bnorm5,
            self.fc1, self.fc2, self.fc3
        ]

    def define_deps(self):
        self.register_dependencies([
            (self.layer1, self.bnorm1, DepType.SAME),
            (self.bnorm1, self.layer2, DepType.INCOMING),
            (self.layer2, self.bnorm2, DepType.SAME),
            (self.bnorm2, self.layer3, DepType.INCOMING),

            (self.layer3, self.bnorm3, DepType.SAME),
            (self.bnorm3, self.layer4, DepType.INCOMING),

            # Link layer 3 to layer 34 and all the additional norm
            (self.layer4, self.bnorm34, DepType.SAME),
            (self.bnorm34, self.layer34, DepType.INCOMING),


            (self.layer34, self.bnorm4, DepType.SAME),
            (self.bnorm4, self.layer5, DepType.INCOMING),

            (self.layer5, self.bnorm5, DepType.SAME),
            (self.bnorm5, self.fc1, DepType.INCOMING),

            (self.fc1, self.fc2, DepType.INCOMING),
            (self.fc2, self.fc3, DepType.INCOMING),
        ])

        self.flatten_conv_id = self.bnorm5.get_module_id()

    def forward(self, x, intermediary: Dict[int, th.Tensor] | None = None):
        self.maybe_update_age(x)
        x = self.layer1(x, intermediary=intermediary)
        x = self.bnorm1(x)
        x = F.relu(x)
        x = self.mpool1(x)

        x = self.layer2(x, intermediary=intermediary)
        x = self.bnorm2(x)
        x = F.relu(x)
        x = self.mpool2(x)

        x = self.layer3(x, intermediary=intermediary)
        x = self.bnorm3(x)
        x = F.relu(x)
        x = self.mpool3(x)
        
        # The newly added layer
        x = self.layer34(x, intermediary=intermediary)
        x = self.bnorm34(x)
        x = F.relu(x)
        x = self.mpool34(x)

        x = self.layer4(x, intermediary=intermediary)
        x = self.bnorm4(x)
        x = F.relu(x)
        x = self.mpool4(x)

        x = self.layer5(x, intermediary=intermediary)
        x = self.bnorm5(x)
        x = F.relu(x)
        x = self.mpool5(x)

        x = x.view(x.size(0), -1)
        x = self.fc1(x, intermediary=intermediary)
        x = F.relu(x)
        x = self.fc2(x, intermediary=intermediary)
        x = F.relu(x)
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


data_dir = "/home/rotaru/Desktop/work/graybox/tiny-224/"
# Normalized & Flipped Dataset
data_transforms = {
    "train": T.Compose(
        [
            T.RandomHorizontalFlip(0.5),
            T.ToTensor(),
            T.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
        ]
    ),
    "val": T.Compose(
        [
            T.ToTensor(),
            T.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
        ]
    ),
    "test": T.Compose(
        [
            T.ToTensor(),
            T.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
        ]
    ),
}
image_datasets = {
    x: ds.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ["train", "val", "test"]
}
device = th.device("cuda:0")


def get_exp():
    model = NanoModel()
    model.define_deps()
    exp = Experiment(
        model=model, optimizer_class=optim.Adam,
        train_dataset=image_datasets["train"],
        eval_dataset=image_datasets["val"],
        device=device, learning_rate=1e-3, batch_size=100,
        training_steps_to_do=200000,
        name="v0",
        root_log_dir='tiny-image-net',
        logger=Dash("tiny-image-net"),
        skip_loading=False)

    def stateful_difference_monitor_callback():
        exp.display_stats()
    return exp