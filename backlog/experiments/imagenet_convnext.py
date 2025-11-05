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

class ConvNeXtBlock(nn.Module):
    def __init__(self, dim, drop=0.0):
        super().__init__()
        self.dwconv = Conv2dWithNeuronOps(dim, dim, kernel_size=3, padding=1, groups=8)
        self.bn = BatchNorm2dWithNeuronOps(dim)
        self.pwconv1 = Conv2dWithNeuronOps(dim, 4*dim, kernel_size=1)
        self.bn1 = BatchNorm2dWithNeuronOps(4*dim)
        self.act = nn.ReLU()
        self.pwconv2 = Conv2dWithNeuronOps(4*dim, dim, kernel_size=1)
        self.bn2 = BatchNorm2dWithNeuronOps(dim)
        self.dropout = nn.Dropout(drop) if drop > 0 else nn.Identity()


    def forward(self, x):
        shortcut = x
        x = self.dwconv(x)
        x = self.bn(x)
        x = self.pwconv1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = self.bn2(x)
        x = self.dropout(x)
        return x + shortcut

class ConvNeXt(NetworkWithOps, nn.Module):
    def __init__(self, num_classes=200, drop=0.2):
        super().__init__()
        self.tracking_mode = TrackingMode.DISABLED

        # dims = [32, 64, 128, 256]
        dims = [8, 16, 32, 64]
        depths = [1, 1, 3, 1]   # blocks per stage

        self.stem = Conv2dWithNeuronOps(3, dims[0], kernel_size=4, stride=4)
        self.stem_bn = BatchNorm2dWithNeuronOps(dims[0])

        # ---- Stage 1 ----
        self.block1_0 = ConvNeXtBlock(dims[0], drop=0.0)
        self.down1 = Conv2dWithNeuronOps(dims[0], dims[1], kernel_size=2, stride=2)
        self.down1_bn = BatchNorm2dWithNeuronOps(dims[1])

        # ---- Stage 2 ----
        self.block2_0 = ConvNeXtBlock(dims[1], drop=0.0)
        self.down2 = Conv2dWithNeuronOps(dims[1], dims[2], kernel_size=2, stride=2)
        self.down2_bn = BatchNorm2dWithNeuronOps(dims[2])

        # ---- Stage 3 ----
        self.block3_0 = ConvNeXtBlock(dims[2], drop=drop)
        self.block3_1 = ConvNeXtBlock(dims[2], drop=drop)
        self.block3_2 = ConvNeXtBlock(dims[2], drop=drop)
        self.down3 = Conv2dWithNeuronOps(dims[2], dims[3], kernel_size=2, stride=2)
        self.down3_bn = BatchNorm2dWithNeuronOps(dims[3])

        # ---- Stage 4 ----
        self.block4_0 = ConvNeXtBlock(dims[3], drop=drop)

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.head_linear = LinearWithNeuronOps(dims[3], 64)
        self.head_act = nn.GELU()    
        self.head_dropout = nn.Dropout(drop)
        self.head_out = LinearWithNeuronOps(64, num_classes)


    def children(self):
        return [
            self.stem, self.stem_bn,
            self.block1_0.dwconv, self.block1_0.bn, self.block1_0.pwconv1, self.block1_0.bn1, self.block1_0.pwconv2, self.block1_0.bn2,
            self.down1, self.down1_bn,
            self.block2_0.dwconv, self.block2_0.bn, self.block2_0.pwconv1, self.block2_0.bn1, self.block2_0.pwconv2, self.block2_0.bn2,
            self.down2, self.down2_bn,
            self.block3_0.dwconv, self.block3_0.bn, self.block3_0.pwconv1, self.block3_0.bn1, self.block3_0.pwconv2, self.block3_0.bn2,
            self.block3_1.dwconv, self.block3_1.bn, self.block3_1.pwconv1, self.block3_1.bn1, self.block3_1.pwconv2, self.block3_1.bn2,
            self.block3_2.dwconv, self.block3_2.bn, self.block3_2.pwconv1, self.block3_2.bn1, self.block3_2.pwconv2, self.block3_2.bn2,
            self.down3, self.down3_bn,
            self.block4_0.dwconv, self.block4_0.bn, self.block4_0.pwconv1, self.block4_0.bn1, self.block4_0.pwconv2, self.block4_0.bn2,
            self.head_linear, self.head_out
        ]
    
    def define_deps(self):
        deps = []
        prev = self.stem_bn

        # Stage 1
        for block in [self.block1_0]:
            deps.extend([
                (prev, block.dwconv, DepType.INCOMING),
                (block.dwconv, block.bn, DepType.SAME),
                (block.bn, block.pwconv1, DepType.INCOMING),
                (block.pwconv1, block.bn1, DepType.SAME),
                (block.bn1, block.pwconv2, DepType.INCOMING),
                (block.pwconv2, block.bn2, DepType.SAME),
            ])
            prev = block.bn2
        deps.extend([
            (prev, self.down1, DepType.INCOMING),
            (self.down1, self.down1_bn, DepType.SAME)
        ])
        prev = self.down1_bn

        # Stage 2
        for block in [self.block2_0]:
            deps.extend([
                (prev, block.dwconv, DepType.INCOMING),
                (block.dwconv, block.bn, DepType.SAME),
                (block.bn, block.pwconv1, DepType.INCOMING),
                (block.pwconv1, block.bn1, DepType.SAME),
                (block.bn1, block.pwconv2, DepType.INCOMING),
                (block.pwconv2, block.bn2, DepType.SAME),
            ])
            prev = block.bn2
        deps.extend([
            (prev, self.down2, DepType.INCOMING),
            (self.down2, self.down2_bn, DepType.SAME)
        ])
        prev = self.down2_bn

        # Stage 3
        for block in [self.block3_0, self.block3_1, self.block3_2]:
            deps.extend([
                (prev, block.dwconv, DepType.INCOMING),
                (block.dwconv, block.bn, DepType.SAME),
                (block.bn, block.pwconv1, DepType.INCOMING),
                (block.pwconv1, block.bn1, DepType.SAME),
                (block.bn1, block.pwconv2, DepType.INCOMING),
                (block.pwconv2, block.bn2, DepType.SAME),
            ])
            prev = block.bn2
        deps.extend([
            (prev, self.down3, DepType.INCOMING),
            (self.down3, self.down3_bn, DepType.SAME)
        ])
        prev = self.down3_bn

        # Stage 4
        for block in [self.block4_0]:
            deps.extend([
                (prev, block.dwconv, DepType.INCOMING),
                (block.dwconv, block.bn, DepType.SAME),
                (block.bn, block.pwconv1, DepType.INCOMING),
                (block.pwconv1, block.bn1, DepType.SAME),
                (block.bn1, block.pwconv2, DepType.INCOMING),
                (block.pwconv2, block.bn2, DepType.SAME),
            ])
            prev = block.bn2

        # Head
        deps.append((prev, self.head_linear, DepType.INCOMING))
        deps.append((self.head_linear, self.head_out, DepType.INCOMING))
        self.register_dependencies(deps)
        self.flatten_conv_id = self.head_linear.get_module_id()

    def forward(self, x):
        self.maybe_update_age(x)
        x = F.relu(self.stem_bn(self.stem(x)))
        x = self.block1_0(x)
        x = F.relu(self.down1_bn(self.down1(x)))

        x = self.block2_0(x)
        x = F.relu(self.down2_bn(self.down2(x)))

        x = self.block3_0(x)
        x = self.block3_1(x)
        x = self.block3_2(x)
        x = F.relu(self.down3_bn(self.down3(x)))

        x = self.block4_0(x)

        x = self.gap(x).view(x.size(0), -1)
        x = self.head_linear(x)
        x = self.head_act(x)        
        x = self.head_dropout(x)
        x = self.head_out(x)
        return x


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

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print('device:', device)


metrics = {
    "acc": MulticlassAccuracy(num_classes=200, average="micro").to(device),
    "f1": MulticlassF1Score(num_classes=200, average="macro").to(device),
}

def get_exp():
    model = ConvNeXt()
    model.define_deps()
    model.to(device)

    exp = Experiment(
        model=model,
        optimizer_class=optim.AdamW,
        train_dataset=train_set,
        eval_dataset=test_set,
        device=device,
        learning_rate=2e-3,
        batch_size=32,
        criterion=nn.CrossEntropyLoss(reduction='none'),
        metrics=metrics,
        training_steps_to_do=500000,
        name="v0",
        root_log_dir="test",
        logger=Dash("test"),
        skip_loading=False
    )
    return exp
