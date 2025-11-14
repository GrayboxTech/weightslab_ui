import os
import sys
_repo_root = os.path.abspath(os.path.dirname(__file__))
_inner_pkg_parent = os.path.join(_repo_root, 'weightslab')
if _inner_pkg_parent not in sys.path:
    sys.path.insert(0, _inner_pkg_parent)

from typing import Dict
import torch as th
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from torchvision import transforms as T
from torchvision import datasets as ds

from weightslab.experiment import Experiment
from weightslab.model_with_ops import NetworkWithOps, DepType
from weightslab.modules_with_ops import (
    Conv2dWithNeuronOps, LinearWithNeuronOps, BatchNorm2dWithNeuronOps
)
from weightslab.tracking import TrackingMode, add_tracked_attrs_to_input_tensor
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score
from experiments.board import Dash

# exported so the trainer can import them
IM_MEAN = (0.1307,)  # MNIST mean
IM_STD  = (0.3081,)  # MNIST std


class FashionCNN(NetworkWithOps):
    def __init__(self):
        super().__init__()
        self.tracking_mode = TrackingMode.DISABLED

        self.layer1 = Conv2dWithNeuronOps(in_channels=1, out_channels=4, kernel_size=3, padding=1)
        self.bnorm1 = BatchNorm2dWithNeuronOps(4)
        self.mpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.layer2 = Conv2dWithNeuronOps(in_channels=4, out_channels=4, kernel_size=3)
        self.bnorm2 = BatchNorm2dWithNeuronOps(4)
        self.mpool2 = nn.MaxPool2d(2)

        self.fc = LinearWithNeuronOps(in_features=4*6*6, out_features=10)

    def children(self):
        return [self.layer1, self.bnorm1, self.layer2, self.bnorm2, self.fc]

    def define_deps(self):
        self.register_dependencies([
            (self.layer1, self.bnorm1, DepType.SAME),
            (self.bnorm1, self.layer2, DepType.INCOMING),
            (self.layer2, self.bnorm2, DepType.SAME),
            (self.bnorm2, self.fc,     DepType.INCOMING),
        ])
        self.flatten_conv_id = self.bnorm2.get_module_id()

    # accept both names because the trainer uses `intermediary_outputs`
    def forward(
        self,
        x,
        intermediary: Dict[int, th.Tensor] | None = None,
        intermediary_outputs: Dict[int, th.Tensor] | None = None
    ):
        if intermediary_outputs is not None and intermediary is None:
            intermediary = intermediary_outputs

        self.maybe_update_age(x)
        orig_x = x  # for tracked attrs

        x = self.layer1(x, intermediary=intermediary)
        x = self.bnorm1(x)
        x = F.relu(x)
        x = self.mpool1(x)

        x = self.layer2(x, intermediary=intermediary)
        x = self.bnorm2(x)
        x = F.relu(x)
        x = self.mpool2(x)

        x = x.view(x.size(0), -1)
        logits = self.fc(x, skip_register=True, intermediary=None)

        one_hot = F.one_hot(logits.argmax(dim=1), num_classes=self.fc.out_features)
        if hasattr(orig_x, 'in_id_batch') and hasattr(orig_x, 'label_batch'):
            add_tracked_attrs_to_input_tensor(
                one_hot,
                in_id_batch=orig_x.in_id_batch,
                label_batch=orig_x.label_batch
            )
        self.fc.register(one_hot)

        return logits  # raw logits (CrossEntropyLoss expects this)


# keep these at module scope (as in your template)
train_set = ds.MNIST("./data", download=True, transform=T.Compose([T.ToTensor()]))
test_set  = ds.MNIST("./data", download=True, train=False, transform=T.Compose([T.ToTensor()]))

device = th.device("cpu")  # or th.device("cuda" if th.cuda.is_available() else "cpu")

metrics = {
    "acc": MulticlassAccuracy(num_classes=10, average="micro").to(device),
    "f1":  MulticlassF1Score(num_classes=10, average="macro").to(device),
}

def get_exp():
    model = FashionCNN()
    model.define_deps()
    exp = Experiment(
        model=model,
        optimizer_class=optim.Adam,
        train_dataset=train_set,
        eval_dataset=test_set,
        device=device,
        learning_rate=1e-3,
        batch_size=100,
        criterion=nn.CrossEntropyLoss(reduction='none'),
        metrics=metrics,
        training_steps_to_do=25600,
        name="v0",
        root_log_dir='test_mnist',
        logger=Dash("test_mnist"),
        skip_loading=False
    )
    return exp
