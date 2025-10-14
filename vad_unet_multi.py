# unet_multitask.py

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets as ds, transforms as T

import torch.optim as optim
from torchmetrics.classification import BinaryAccuracy

from weightslab.experiment import Experiment
from weightslab.tasks import Task
from weightslab.model_with_ops import NetworkWithOps, DepType
from weightslab.modules_with_ops import Conv2dWithNeuronOps, LinearWithNeuronOps, BatchNorm2dWithNeuronOps
from weightslab.tracking import TrackingMode
from board import Dash

class UNetMulti(NetworkWithOps, nn.Module):
    def __init__(self, in_ch: int = 3, base: int = 4, bottleneck: int = 32):
        super().__init__()
        self.tracking_mode = TrackingMode.DISABLED
        C1, C2, C3 = base, base * 2, bottleneck

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

        # ---- Decoder ----
        self.up1_conv = Conv2dWithNeuronOps(C3 + C2, C2, kernel_size=3, padding=1)
        self.up1_bn   = BatchNorm2dWithNeuronOps(C2)

        self.up2_conv = Conv2dWithNeuronOps(C2 + C1, C1, kernel_size=3, padding=1)
        self.up2_bn   = BatchNorm2dWithNeuronOps(C1)

        # ---- Heads ----
        self.seg_feats   = Conv2dWithNeuronOps(C1, C1, kernel_size=1)   # shared feat
        self.classifier  = LinearWithNeuronOps(C1, 1)                   # classification
        self.recon_head  = Conv2dWithNeuronOps(C1, in_ch, kernel_size=1)  # reconstruction

    def children(self):
        return [
            self.enc1_conv, self.enc1_bn,
            self.down1_conv, self.down1_bn,
            self.enc2_conv, self.enc2_bn,
            self.down2_conv, self.down2_bn,
            self.mid_conv5, self.mid_bn5,
            self.mid_conv7, self.mid_bn7,
            self.up1_conv, self.up1_bn,
            self.up2_conv, self.up2_bn,
            self.seg_feats, self.classifier, self.recon_head
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
        ])

    def forward(self, x, intermediary_outputs=None):
        self.maybe_update_age(x)

        # ---- Encoder ----
        e1 = F.relu(self.enc1_bn(self.enc1_conv(x, intermediary=intermediary_outputs)))
        d1 = F.relu(self.down1_bn(self.down1_conv(e1, intermediary=intermediary_outputs)))

        e2 = F.relu(self.enc2_bn(self.enc2_conv(d1, intermediary=intermediary_outputs)))
        d2 = F.relu(self.down2_bn(self.down2_conv(e2, intermediary=intermediary_outputs)))

        # ---- Bottleneck ----
        m  = F.relu(self.mid_bn5(self.mid_conv5(d2, intermediary=intermediary_outputs)))
        m  = F.relu(self.mid_bn7(self.mid_conv7(m, intermediary=intermediary_outputs)))

        # ---- Decoder ----
        u1 = F.interpolate(m, size=e2.shape[-2:], mode='bilinear', align_corners=False)
        u1 = torch.cat([u1, e2], dim=1)
        u1 = F.relu(self.up1_bn(self.up1_conv(u1, intermediary=intermediary_outputs)))

        u2 = F.interpolate(u1, size=e1.shape[-2:], mode='bilinear', align_corners=False)
        u2 = torch.cat([u2, e1], dim=1)
        u2 = F.relu(self.up2_bn(self.up2_conv(u2, intermediary=intermediary_outputs)))

        return u2  # feature map

    def forward_head(self, name: str, x: torch.Tensor):
        feats = self.forward(x)
        if name == "class":
            pooled = feats.mean(dim=(2, 3))  # global avg pool
            return self.classifier(pooled).squeeze(1)
        elif name == "recon":
            return self.recon_head(feats)    # image reconstruction
        else:
            raise KeyError(name)

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

print("train classes:", train_dataset.class_to_idx)
print("val classes:", val_dataset.class_to_idx)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_exp():
    model = UNetMulti(in_ch=3, base=4, bottleneck=32)
    model.define_deps()
    model.to(device)

    # tasks
    cls_task = Task(
        name="class",
        model=model,
        criterion=nn.BCEWithLogitsLoss(reduction="none"),
        loss_weight=1.0,
        target_fn=lambda inp: inp.label_batch.float(),
        metrics={"acc": BinaryAccuracy().to(device)}
    )

    recon_task = Task(
        name="recon",
        model=model,
        criterion=nn.MSELoss(reduction="none"),
        loss_weight=0.3,
        target_fn=lambda inp: inp
    )

    exp = Experiment(
        model=model,
        optimizer_class=optim.Adam,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        device=device,
        learning_rate=2e-4,
        batch_size=64,
        training_steps_to_do=200000,
        name="vad_unet_multitask",
        root_log_dir="test_vad_unet_multitask",
        logger=Dash("test_vad_unet_multitask"),
        tasks=[cls_task, recon_task],  
        skip_loading=False
    )
    return exp


if __name__ == "__main__":
    exp = get_exp()
    exp.set_is_training(True)
    exp.train_step_or_eval_full()
