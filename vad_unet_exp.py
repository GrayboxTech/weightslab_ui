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

class UNet(NetworkWithOps, nn.Module):
    def __init__(self, in_ch: int = 3, base: int = 4, bottleneck: int = 32, return_segmentation: bool = False):
        super().__init__()
        self.tracking_mode = TrackingMode.DISABLED
        self.return_segmentation = return_segmentation

        C1, C2, C3 = base, base * 2, bottleneck  # 4, 8, 32

        # ---- Encoder ----
        self.enc1_conv = Conv2dWithNeuronOps(in_ch, C1, kernel_size=3, padding=1)
        self.enc1_bn   = BatchNorm2dWithNeuronOps(C1)

        self.down1_conv = Conv2dWithNeuronOps(C1, C2, kernel_size=3, stride=2, padding=1)
        self.down1_bn   = BatchNorm2dWithNeuronOps(C2)

        self.enc2_conv = Conv2dWithNeuronOps(C2, C2, kernel_size=3, padding=1)
        self.enc2_bn   = BatchNorm2dWithNeuronOps(C2)

        self.down2_conv = Conv2dWithNeuronOps(C2, C3, kernel_size=3, stride=2, padding=1)
        self.down2_bn   = BatchNorm2dWithNeuronOps(C3)

        # ---- Bottleneck ----
        self.mid_conv5  = Conv2dWithNeuronOps(C3, C3, kernel_size=5, padding=2)
        self.mid_bn5    = BatchNorm2dWithNeuronOps(C3)
        self.mid_conv7  = Conv2dWithNeuronOps(C3, C3, kernel_size=7, padding=3)
        self.mid_bn7    = BatchNorm2dWithNeuronOps(C3)

        # ---- Decoder  ----
        self.up1_conv = Conv2dWithNeuronOps(C3 + C2, C2, kernel_size=3, padding=1)
        self.up1_bn   = BatchNorm2dWithNeuronOps(C2)

        self.up2_conv = Conv2dWithNeuronOps(C2 + C1, C1, kernel_size=3, padding=1)
        self.up2_bn   = BatchNorm2dWithNeuronOps(C1)

        # ---- Heads ----
        self.seg_head   = Conv2dWithNeuronOps(C1, 1, kernel_size=1, padding=0)
        self.classifier = LinearWithNeuronOps(1, 1)

    def children(self):
        return [
            self.enc1_conv, self.enc1_bn,
            self.down1_conv, self.down1_bn,
            self.enc2_conv, self.enc2_bn,
            self.down2_conv, self.down2_bn,
            self.mid_conv5, self.mid_bn5, self.mid_conv7, self.mid_bn7,
            self.up1_conv, self.up1_bn,
            self.up2_conv, self.up2_bn,
            self.seg_head, self.classifier
        ]

    def define_deps(self):
        self.register_dependencies([
            (self.enc1_conv, self.enc1_bn, DepType.SAME),
            (self.enc1_bn,   self.down1_conv, DepType.INCOMING),

            (self.down1_conv, self.down1_bn, DepType.SAME),
            (self.down1_bn,   self.enc2_conv, DepType.INCOMING),
            (self.enc2_conv,  self.enc2_bn, DepType.SAME),
            (self.enc2_bn,    self.down2_conv, DepType.INCOMING),

            (self.down2_conv, self.down2_bn, DepType.SAME),
            (self.down2_bn,   self.mid_conv5, DepType.INCOMING),
            (self.mid_conv5,  self.mid_bn5, DepType.SAME),
            (self.mid_bn5,    self.mid_conv7, DepType.INCOMING),
            (self.mid_conv7,  self.mid_bn7, DepType.SAME),

            (self.mid_bn7,    self.up1_conv, DepType.INCOMING),
            (self.up1_conv,   self.up1_bn, DepType.SAME),

            (self.up1_bn,     self.up2_conv, DepType.INCOMING),
            (self.up2_conv,   self.up2_bn, DepType.SAME),

            (self.up2_bn,     self.seg_head, DepType.INCOMING),
            (self.seg_head,   self.classifier, DepType.INCOMING),
        ])
        self.flatten_conv_id = self.up2_bn.get_module_id()

    def _relu(self, x): return F.relu(x)

    def forward(self, x, intermediary_outputs=None):
        self.maybe_update_age(x)

        # ---- Encoder ----
        e1 = self._relu(self.enc1_bn(self.enc1_conv(x, intermediary=intermediary_outputs)))
        d1 = self._relu(self.down1_bn(self.down1_conv(e1, intermediary=intermediary_outputs)))

        e2 = self._relu(self.enc2_bn(self.enc2_conv(d1, intermediary=intermediary_outputs)))
        d2 = self._relu(self.down2_bn(self.down2_conv(e2, intermediary=intermediary_outputs)))

        # ---- Bottleneck ----
        m  = self._relu(self.mid_bn5(self.mid_conv5(d2, intermediary=intermediary_outputs)))
        m  = self._relu(self.mid_bn7(self.mid_conv7(m,  intermediary=intermediary_outputs)))

        # ---- Decoder ----
        u1 = F.interpolate(m,  size=e2.shape[-2:], mode='bilinear', align_corners=False)
        u1 = th.cat([u1, e2], dim=1)
        u1 = self._relu(self.up1_bn(self.up1_conv(u1, intermediary=intermediary_outputs)))

        u2 = F.interpolate(u1, size=e1.shape[-2:], mode='bilinear', align_corners=False)
        u2 = th.cat([u2, e1], dim=1)
        u2 = self._relu(self.up2_bn(self.up2_conv(u2, intermediary=intermediary_outputs)))

        seg_logits = self.seg_head(u2, intermediary=intermediary_outputs)  # [B,1,H,W]

        if self.return_segmentation:
            return seg_logits.squeeze(1)

        pooled = seg_logits.mean(dim=(2, 3))  # [B,1]
        out = self.classifier(pooled)         # [B,1]
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

train_dataset = ds.ImageFolder(os.path.join(root_dir, "train"), transform=train_transform)
val_dataset   = ds.ImageFolder(os.path.join(root_dir, "test"),  transform=val_transform)

print(train_dataset.class_to_idx) 
print(val_dataset.class_to_idx) 

device = th.device("cuda" if th.cuda.is_available() else "cpu")

metrics = {
    "acc": BinaryAccuracy().to(device),
    # "f1": BinaryF1Score().to(device),
}


def get_exp():
    model = UNet(in_ch=3, base=4, bottleneck=32, return_segmentation=False)
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
        root_log_dir='test_vad_unet',
        logger=Dash("test_vad_unet"),
        criterion=nn.BCEWithLogitsLoss(reduction='none'),
        skip_loading=False
    )
    return exp


if __name__ == "__main__":
    exp = get_exp()
    exp.train_step_or_eval_full()
