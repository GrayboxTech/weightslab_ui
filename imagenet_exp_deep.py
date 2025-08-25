import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
import torchvision.transforms as tt
from torchvision.datasets import ImageFolder
from torchmetrics.classification import MulticlassAccuracy
from weightslab.experiment import Experiment
from weightslab.model_with_ops import NetworkWithOps, DepType
from weightslab.modules_with_ops import (
    Conv2dWithNeuronOps,
    LinearWithNeuronOps,
    BatchNorm2dWithNeuronOps
)
from weightslab.tracking import TrackingMode, add_tracked_attrs_to_input_tensor
from board import Dash


class ConvBlockWithOps(nn.Module):
    def __init__(self, in_ch, out_ch, pool=True):
        super().__init__()
        self.conv1 = Conv2dWithNeuronOps(in_ch,  out_ch, kernel_size=3, stride=1, padding=1)
        self.bn1   = BatchNorm2dWithNeuronOps(out_ch)
        self.conv2 = Conv2dWithNeuronOps(out_ch, out_ch, kernel_size=3, stride=1, padding=1)
        self.bn2   = BatchNorm2dWithNeuronOps(out_ch)
        self.pool = pool

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        if self.pool:
            x = F.max_pool2d(x, 2)
        return x


class TinyImageNet_2(NetworkWithOps, nn.Module):
    def __init__(self):
        super().__init__()
        self.tracking_mode = TrackingMode.DISABLED

        # 64x64 input
        self.b1 = ConvBlockWithOps(3,    8,  pool=True)   # 64 → 32
        self.b2 = ConvBlockWithOps(8,    16, pool=True)   # 32 → 16
        self.b3 = ConvBlockWithOps(16,   32, pool=True)   # 16 → 8
        self.b4 = ConvBlockWithOps(32,   64, pool=True)   # 8  → 4
        self.b5 = ConvBlockWithOps(64,   128, pool=False) # 4×4

        self.gap     = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=0.1)

        self.fc1 = LinearWithNeuronOps(128, 256)
        self.fc2 = LinearWithNeuronOps(256, 200)

    def children(self):
        return [
            self.b1.conv1, self.b1.bn1, self.b1.conv2, self.b1.bn2,
            self.b2.conv1, self.b2.bn1, self.b2.conv2, self.b2.bn2,
            self.b3.conv1, self.b3.bn1, self.b3.conv2, self.b3.bn2,
            self.b4.conv1, self.b4.bn1, self.b4.conv2, self.b4.bn2,
            self.b5.conv1, self.b5.bn1, self.b5.conv2, self.b5.bn2,
            self.fc1, self.fc2
        ]

    def define_deps(self):
        regs = []
        blocks = [self.b1, self.b2, self.b3, self.b4, self.b5]

        # per-block SAME/INCOMING
        for b in blocks:
            regs += [
                (b.conv1, b.bn1,  DepType.SAME),
                (b.bn1,   b.conv2, DepType.INCOMING),
                (b.conv2, b.bn2,  DepType.SAME),
            ]

        for prev, nxt in zip(blocks[:-1], blocks[1:]):
            regs.append((prev.bn2, nxt.conv1, DepType.INCOMING))

        regs.append((self.b5.bn2, self.fc1, DepType.INCOMING))
        regs.append((self.fc1,    self.fc2, DepType.INCOMING))

        self.register_dependencies(regs)

    def forward(self, x):
        self.maybe_update_age(x)

        x = self.b1(x)  # 64→32
        x = self.b2(x)  # 32→16
        x = self.b3(x)  # 16→8
        x = self.b4(x)  # 8 →4
        x = self.b5(x)  # 4×4

        x = self.gap(x)               # → 1×1
        x = x.view(x.size(0), -1)     # (B, 128)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))

        output = self.fc2(x, skip_register=True, intermediary=None)

        one_hot = F.one_hot(output.argmax(dim=1), num_classes=self.fc2.out_features)
        if hasattr(x, 'in_id_batch') and hasattr(x, 'label_batch'):
            add_tracked_attrs_to_input_tensor(
                one_hot, in_id_batch=x.in_id_batch, label_batch=x.label_batch
            )
        self.fc2.register(one_hot)
        return output


mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]

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
test_set  = ImageFolder('./data/tiny-imagenet-200/val/classified', transform=transform_test)

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print('device:', device)

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
        root_log_dir="tinyimagenet_deep_exp0",
        logger=Dash("tinyimagenet_deep_exp0"),
        skip_loading=False
    )
    return exp

if __name__ == "__main__":
    exp = get_exp()
    exp.set_is_training(True)
    for step in range(10):
        exp.train_step_or_eval_full()
        print(f'Step {step+1}')
