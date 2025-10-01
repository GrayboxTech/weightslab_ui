# vad-unet-reconstruction_autoencoder.py

import os
from typing import List
import torch
import torch as th
import torch.nn as nn
import torch.optim as optim

from torch.nn import functional as F
from torchvision import transforms as T
from torchvision.datasets.folder import default_loader, IMG_EXTENSIONS

from torch.utils.data import Dataset, ConcatDataset

from weightslab.experiment import Experiment
from weightslab.model_with_ops import NetworkWithOps, DepType
from weightslab.modules_with_ops import (
    Conv2dWithNeuronOps, BatchNorm2dWithNeuronOps, LinearWithNeuronOps
)
from weightslab.tracking import TrackingMode
from board import Dash


class UNetAE(NetworkWithOps, nn.Module):
    def __init__(self, in_ch: int = 3, base: int = 4, bottleneck: int = 32):
        super().__init__()
        self.tracking_mode = TrackingMode.DISABLED

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

        # ---- Decoder ----
        self.up1_conv = Conv2dWithNeuronOps(C3 + C2, C2, kernel_size=3, padding=1)
        self.up1_bn   = BatchNorm2dWithNeuronOps(C2)

        self.up2_conv = Conv2dWithNeuronOps(C2 + C1, C1, kernel_size=3, padding=1)
        self.up2_bn   = BatchNorm2dWithNeuronOps(C1)

        # ---- Reconstruction head (RGB) ----
        self.recon_head = Conv2dWithNeuronOps(C1, in_ch, kernel_size=1, padding=0)
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
            self.recon_head, self.classifier
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

            (self.up2_bn,     self.recon_head, DepType.INCOMING),
        ])
        self.flatten_conv_id = self.up2_bn.get_module_id()

    def _relu(self, x): return F.relu(x)

    def forward(self, x, intermediary_outputs=None):
        self.maybe_update_age(x)

        # encoder
        e1 = self._relu(self.enc1_bn(self.enc1_conv(x, intermediary=intermediary_outputs)))
        d1 = self._relu(self.down1_bn(self.down1_conv(e1, intermediary=intermediary_outputs)))

        e2 = self._relu(self.enc2_bn(self.enc2_conv(d1, intermediary=intermediary_outputs)))
        d2 = self._relu(self.down2_bn(self.down2_conv(e2, intermediary=intermediary_outputs)))

        # bottleneck
        m  = self._relu(self.mid_bn5(self.mid_conv5(d2, intermediary=intermediary_outputs)))
        m  = self._relu(self.mid_bn7(self.mid_conv7(m,  intermediary=intermediary_outputs)))

        # decoder
        u1 = F.interpolate(m,  size=e2.shape[-2:], mode='bilinear', align_corners=False)
        u1 = th.cat([u1, e2], dim=1)
        u1 = self._relu(self.up1_bn(self.up1_conv(u1, intermediary=intermediary_outputs)))

        u2 = F.interpolate(u1, size=e1.shape[-2:], mode='bilinear', align_corners=False)
        u2 = th.cat([u2, e1], dim=1)
        u2 = self._relu(self.up2_bn(self.up2_conv(u2, intermediary=intermediary_outputs)))

        # reconstruction
        out = self.recon_head(u2, intermediary=intermediary_outputs)  # [B,3,H,W]
        return out


IM_MEAN = (.5533, .5829, .5946)
IM_STD  = (.1527, .1628, .1726)

img_transform = T.Compose([
    T.ToTensor(),
    T.Normalize(IM_MEAN, IM_STD),
])

class SelfReconstructionFolder(Dataset):
    def __init__(self, folders, image_size=None):
        if isinstance(folders, str): folders = [folders]
        self.files = []
        self.image_size = image_size
        for fd in folders:
            if not os.path.isdir(fd): continue
            for name in sorted(os.listdir(fd)):
                if any(name.lower().endswith(ext) for ext in IMG_EXTENSIONS):
                    self.files.append(os.path.join(fd, name))
        if len(self.files) == 0:
            raise RuntimeError(f"No images found in {folders}")
        self.loader = default_loader
        self.images = self.files 

    def __len__(self): return len(self.files)

    def __getitem__(self, idx):
        img = self.loader(self.files[idx])  # PIL
        if self.image_size is not None:
            img = T.functional.resize(img, self.image_size, interpolation=T.InterpolationMode.BILINEAR)
        x = img_transform(img)              # [3,H,W]
        return x, x                         # target is the same tensor

root_dir = "VAD"

train_folders = [os.path.join(root_dir, "train", "good")]
train_dataset = SelfReconstructionFolder(train_folders, image_size=None)

eval_folders = [
    os.path.join(root_dir, "test", "good"),
    os.path.join(root_dir, "test", "bad"),
]
eval_dataset = SelfReconstructionFolder(eval_folders, image_size=None)

device = th.device("cuda" if th.cuda.is_available() else "cpu")

metrics = {} 


def get_exp():
    model = UNetAE(in_ch=3, base=4, bottleneck=32)
    model.define_deps()
    model.to(device)

    exp = Experiment(
        model=model,
        optimizer_class=optim.Adam,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        device=device,
        learning_rate=2e-4,
        batch_size=32,
        training_steps_to_do=200000,
        name="recon_autoencoder_v0",
        metrics=metrics,
        root_log_dir="vad_unet_recon",
        logger=Dash("vad_unet_recon"),
        criterion=nn.L1Loss(reduction="mean"),
        skip_loading=False,
        task_type="reconstruction"  
    )
    return exp


if __name__ == "__main__":
    exp = get_exp()
    exp.train_step_or_eval_full()
