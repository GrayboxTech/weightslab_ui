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
from weightslab.tracking import TrackingMode, add_tracked_attrs_to_input_tensor

from board import Dash

# class TinyImageNetNet(NetworkWithOps, nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.tracking_mode = TrackingMode.DISABLED

#         self.conv1 = Conv2dWithNeuronOps(3,   8, kernel_size=3, padding=1)
#         self.bn1   = BatchNorm2dWithNeuronOps(8)

#         self.conv2 = Conv2dWithNeuronOps(8, 16, kernel_size=3, padding=1)
#         self.bn2   = BatchNorm2dWithNeuronOps(16)

#         self.conv3 = Conv2dWithNeuronOps(16, 32, kernel_size=3, padding=1)
#         self.bn3   = BatchNorm2dWithNeuronOps(32)

#         self.conv4 = Conv2dWithNeuronOps(32, 64, kernel_size=3, padding=1)
#         self.bn4   = BatchNorm2dWithNeuronOps(64)

#         self.conv5 = Conv2dWithNeuronOps(64, 128, kernel_size=3, padding=1)
#         self.bn5   = BatchNorm2dWithNeuronOps(128)

#         self.pool = nn.MaxPool2d(2)                 # down-sample ×2
#         self.gap  = nn.AdaptiveAvgPool2d(1)         # 4×4 → 1×1
#         self.dropout = nn.Dropout(p=0.20)

#         self.fc1 = LinearWithNeuronOps(128, 128)
#         self.fc2 = LinearWithNeuronOps(128, 256)
#         self.fc3 = LinearWithNeuronOps(256, 200)

#         self.softmax = nn.Softmax(dim=1)


#     def children(self):
#         return [
#             self.conv1, self.bn1,
#             self.conv2, self.bn2,
#             self.conv3, self.bn3,
#             self.conv4, self.bn4,
#             self.conv5, self.bn5,
#             self.fc1, self.fc2, self.fc3
#         ]

#     def define_deps(self):
#         self.register_dependencies([
#             (self.conv1, self.bn1, DepType.SAME),
#             (self.bn1,   self.conv2, DepType.INCOMING),
#             (self.conv2, self.bn2, DepType.SAME),
#             (self.bn2,   self.conv3, DepType.INCOMING),
#             (self.conv3, self.bn3, DepType.SAME),
#             (self.bn3,   self.conv4, DepType.INCOMING),
#             (self.conv4, self.bn4, DepType.SAME),
#             (self.bn4,   self.conv5, DepType.INCOMING),
#             (self.conv5, self.bn5, DepType.SAME),
#             (self.bn5,   self.fc1,   DepType.INCOMING),
#             (self.fc1,   self.fc2,   DepType.INCOMING),
#             (self.fc2,   self.fc3,   DepType.INCOMING)
#         ])
#         self.flatten_conv_id = self.bn3.get_module_id()

#     def forward(self, x):
#         self.maybe_update_age(x)
#         x = self.pool(F.relu(self.bn1(self.conv1(x))))  # 64→32
#         x = self.pool(F.relu(self.bn2(self.conv2(x))))  # 32→16
#         x = self.pool(F.relu(self.bn3(self.conv3(x))))  # 16→8
#         x = self.pool(F.relu(self.bn4(self.conv4(x))))  # 8→4
#         x = self.pool(F.relu(self.bn5(self.conv5(x))))  # 4→2
#         x = self.gap(x)                                 # 2x2 → 1x1
#         x = x.view(x.size(0), -1)                       # (B, 128)
#         x = self.dropout(x)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))

#         output = self.fc3(x, skip_register=True, intermediary=None)

#         one_hot = F.one_hot(output.argmax(dim=1), num_classes=self.fc3.out_features)

#         if hasattr(x, 'in_id_batch') and hasattr(x, 'label_batch'):
#             add_tracked_attrs_to_input_tensor(
#                 one_hot, in_id_batch=x.in_id_batch, label_batch=x.label_batch)

#         self.fc3.register(one_hot)
#         output = self.softmax(output)
#         return output
    

class TinyImageNet_2(NetworkWithOps, nn.Module):
    def __init__(self):
        super().__init__()
        self.tracking_mode = TrackingMode.DISABLED

        self.conv1 = Conv2dWithNeuronOps(3,   16, kernel_size=3, padding=1)
        self.bn1   = BatchNorm2dWithNeuronOps(16)

        self.conv2 = Conv2dWithNeuronOps(16, 32, kernel_size=3, padding=1)
        self.bn2   = BatchNorm2dWithNeuronOps(32)

        self.conv3 = Conv2dWithNeuronOps(32, 64, kernel_size=3, padding=1)
        self.bn3   = BatchNorm2dWithNeuronOps(64)

        self.conv4 = Conv2dWithNeuronOps(64, 128, kernel_size=3, padding=1)
        self.bn4   = BatchNorm2dWithNeuronOps(128)

        self.conv5 = Conv2dWithNeuronOps(128, 256, kernel_size=3, padding=1)
        self.bn5   = BatchNorm2dWithNeuronOps(256)

        self.pool = nn.MaxPool2d(2)
        self.gap  = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=0.3)

        self.fc1 = LinearWithNeuronOps(256, 512)
        self.fc2 = LinearWithNeuronOps(512, 200)

        self.softmax = nn.Softmax(dim=1)

    def children(self):
        return [self.conv1, self.bn1,
                self.conv2, self.bn2,
                self.conv3, self.bn3,
                self.conv4, self.bn4,
                self.conv5, self.bn5,
                self.fc1, self.fc2]

    def define_deps(self):
        self.register_dependencies([
            (self.conv1, self.bn1, DepType.SAME),
            (self.bn1,   self.conv2, DepType.INCOMING),
            (self.conv2, self.bn2, DepType.SAME),
            (self.bn2,   self.conv3, DepType.INCOMING),
            (self.conv3, self.bn3, DepType.SAME),
            (self.bn3,   self.conv4, DepType.INCOMING),
            (self.conv4, self.bn4, DepType.SAME),
            (self.bn4,   self.conv5, DepType.INCOMING),
            (self.conv5, self.bn5, DepType.SAME),
            (self.bn5,   self.fc1,   DepType.INCOMING),
            (self.fc1,   self.fc2,   DepType.INCOMING),
        ])
        self.flatten_conv_id = self.bn3.get_module_id()

    def forward(self, x):
        self.maybe_update_age(x)
        for conv, bn in [(self.conv1, self.bn1),
                         (self.conv2, self.bn2),
                         (self.conv3, self.bn3),
                         (self.conv4, self.bn4),
                         (self.conv5, self.bn5)]:
            x = self.pool(F.relu(bn(conv(x))))
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        output = self.fc2(x, skip_register=True, intermediary=None)

        one_hot = F.one_hot(output.argmax(dim=1), num_classes=self.fc2.out_features)

        if hasattr(x, 'in_id_batch') and hasattr(x, 'label_batch'):
            add_tracked_attrs_to_input_tensor(
                one_hot, in_id_batch=x.in_id_batch, label_batch=x.label_batch)

        self.fc2.register(one_hot)


        return output

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

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print('device:', device)

def custom_top5_accuracy(output, labels):
    top5 = output.topk(5, dim=1).indices
    correct = top5.eq(labels.view(-1, 1)).sum().item()
    return correct / labels.size(0)

metrics = {
    "acc": MulticlassAccuracy(num_classes=200, average="micro").to(device),
}


def get_exp():
    model = TinyImageNet_2()
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
        name="v0",
        root_log_dir="tinyimagenet-exp2_ckpt",
        logger=Dash("tinyimagenet-exp2_ckpt"),
        skip_loading=False
    )

    return exp
