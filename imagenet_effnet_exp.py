import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as tt
from torchvision.datasets import ImageFolder
from torchvision.models import efficientnet_b0
from torch.nn import functional as F

from weightslab.experiment import Experiment
from weightslab.model_with_ops import NetworkWithOps
from weightslab.tracking import TrackingMode
from board import Dash

# TinyImageNet normalization (same as ImageNet)
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

transform_train = tt.Compose([
    tt.Resize(256),
    tt.CenterCrop(224),
    tt.RandomHorizontalFlip(),
    tt.ToTensor(),
    tt.Normalize(mean, std)
])

transform_test = tt.Compose([
    tt.Resize(256),
    tt.CenterCrop(224),
    tt.ToTensor(),
    tt.Normalize(mean, std)
])

train_set = ImageFolder('./data/tiny-imagenet-200/train', transform=transform_train)
test_set = ImageFolder('./data/tiny-imagenet-200/val/classified', transform=transform_test)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EfficientNetTinyINet(NetworkWithOps, nn.Module):
    def __init__(self):
        super().__init__()
        self.tracking_mode = TrackingMode.DISABLED
        self.backbone = efficientnet_b0(weights=None)
        in_feats = self.backbone.classifier[1].in_features
        self.backbone.classifier[1] = nn.Linear(in_feats, 200)

    def children(self):
        return [self.backbone]

    def define_deps(self):
        # No custom dependencies
        pass

    def forward(self, x):
        self.maybe_update_age(x)
        return self.backbone(x)


def get_exp():
    model = EfficientNetTinyINet()
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
        name="tinyimagenet_effnet_b0",
        root_log_dir="tinyimagenet-effnet-exp0",
        logger=Dash("tinyimagenet-effnet-exp0"),
        skip_loading=False,
    )

    exp.register_train_loop_callback(lambda: exp.display_stats())
    exp.set_train_loop_clbk_freq(10)

    return exp
