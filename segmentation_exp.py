import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision.datasets import VOCSegmentation
import torchvision.transforms as T
from torchmetrics.classification import MulticlassJaccardIndex
from weightslab.experiment import Experiment
from weightslab.model_with_ops import NetworkWithOps, DepType
from weightslab.modules_with_ops import (
    Conv2dWithNeuronOps, LinearWithNeuronOps, BatchNorm2dWithNeuronOps
)
from weightslab.tracking import TrackingMode
from board import Dash
from io import BytesIO
from PIL import Image
import base64
from torch.utils.data import DataLoader

DATA_ROOT = './data'
VOC_YEAR = '2012'
IMG_SIZE = (256, 256)
NUM_CLASSES = 21

def mask_to_png_bytes(mask):
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()
    mask = (mask.astype(np.uint8))
    im = Image.fromarray(mask)
    buf = BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()

class DualTransform:
    def __init__(self, img_transform, mask_transform):
        self.img_transform = img_transform
        self.mask_transform = mask_transform
    def __call__(self, img, mask):
        img = self.img_transform(img)
        mask = self.mask_transform(mask)
        return img, mask

img_transform = T.Compose([
    T.Resize(IMG_SIZE, interpolation=T.InterpolationMode.BILINEAR),
    T.ToTensor(),
])
mask_transform = T.Compose([
    T.Resize(IMG_SIZE, interpolation=T.InterpolationMode.NEAREST),
    T.Lambda(lambda m: torch.as_tensor(np.array(m), dtype=torch.long))
])
dual_transform = DualTransform(img_transform, mask_transform)

class VOCSegmentationWithIndex(VOCSegmentation):
    def __getitem__(self, idx):
        img, mask = super().__getitem__(idx)
        return img, mask, int(idx)
    
class UNet(NetworkWithOps, nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.tracking_mode = TrackingMode.DISABLED
        # Encoder
        self.enc1 = Conv2dWithNeuronOps(3, 64, kernel_size=3, padding=1)
        self.bn1 = BatchNorm2dWithNeuronOps(64)
        self.enc2 = Conv2dWithNeuronOps(64, 128, kernel_size=3, padding=1)
        self.bn2 = BatchNorm2dWithNeuronOps(128)
        self.enc3 = Conv2dWithNeuronOps(128, 256, kernel_size=3, padding=1)
        self.bn3 = BatchNorm2dWithNeuronOps(256)
        self.pool = nn.MaxPool2d(2)

        # Decoder
        self.dec2 = Conv2dWithNeuronOps(256, 128, kernel_size=3, padding=1)
        self.bn4 = BatchNorm2dWithNeuronOps(128)
        self.dec1 = Conv2dWithNeuronOps(128, 64, kernel_size=3, padding=1)
        self.bn5 = BatchNorm2dWithNeuronOps(64)

        self.outc = Conv2dWithNeuronOps(64, num_classes, kernel_size=1)

    def children(self):
        return [
            self.enc1, self.bn1, self.enc2, self.bn2, self.enc3, self.bn3,
            self.dec2, self.bn4, self.dec1, self.bn5, self.outc
        ]

    def define_deps(self):
        self.register_dependencies([
            (self.enc1, self.bn1, DepType.SAME),
            (self.bn1, self.enc2, DepType.INCOMING),
            (self.enc2, self.bn2, DepType.SAME),
            (self.bn2, self.enc3, DepType.INCOMING),
            (self.enc3, self.bn3, DepType.SAME),
            (self.bn3, self.dec2, DepType.INCOMING),
            (self.dec2, self.bn4, DepType.SAME),
            (self.bn4, self.dec1, DepType.INCOMING),
            (self.dec1, self.bn5, DepType.SAME),
            (self.bn5, self.outc, DepType.INCOMING)
        ])
        self.flatten_conv_id = self.bn5.get_module_id()

    def forward(self, x):
        self.maybe_update_age(x)
        x1 = torch.relu(self.bn1(self.enc1(x)))        # [B, 64, H, W]
        x2 = self.pool(x1)                             # [B, 64, H/2, W/2]
        x2 = torch.relu(self.bn2(self.enc2(x2)))       # [B, 128, H/2, W/2]
        x3 = self.pool(x2)                             # [B, 128, H/4, W/4]
        x3 = torch.relu(self.bn3(self.enc3(x3)))       # [B, 256, H/4, W/4]

        # Decoder
        d2 = nn.functional.interpolate(x3, scale_factor=2, mode='bilinear', align_corners=False)   # [B, 256, H/2, W/2]
        d2 = torch.relu(self.bn4(self.dec2(d2)))
        d2 = d2 + x2 

        d1 = nn.functional.interpolate(d2, scale_factor=2, mode='bilinear', align_corners=False)   # [B, 128, H, W]
        d1 = torch.relu(self.bn5(self.dec1(d1)))
        d1 = d1 + x1  

        out = self.outc(d1)    # [B, num_classes, H, W]
        return out

train_dataset = VOCSegmentationWithIndex(
    root=DATA_ROOT, year=VOC_YEAR, image_set='train', download=True, transforms=dual_transform
)

img, mask, idx = train_dataset[0]
print(np.unique(mask.numpy()))

val_dataset = VOCSegmentationWithIndex(
    root=DATA_ROOT, year=VOC_YEAR, image_set='val', download=True, transforms=dual_transform
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


tmp_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
for batch in tmp_loader:
    img, mask, idx = batch 
    break 
task_config = {
    "type": "segmentation",
    "num_classes": NUM_CLASSES,
    "mask_format": "png"
}

metrics = {
    "mean_iou": MulticlassJaccardIndex(num_classes=NUM_CLASSES, ignore_index=255)
}

def cross_entropy_2d(inputs, targets, ignore_index=255):
    assert targets.ndim == 3, f"Expected 3D mask batch, got {targets.shape}"
    loss = nn.functional.cross_entropy(inputs, targets, ignore_index=ignore_index, reduction='none')
    return loss.mean(dim=[1,2]) 


def get_exp():
    model = UNet(NUM_CLASSES)
    model.define_deps()
    model.to(device)
    exp = Experiment(
        model=model,
        optimizer_class=optim.Adam,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        device=device,
        learning_rate=1e-3,
        batch_size=8,
        criterion=nn.CrossEntropyLoss(ignore_index=255),
        metrics=metrics,
        training_steps_to_do=25600,
        name="voc2012-seg-test2",
        root_log_dir="voc2012-seg-test2",
        logger=Dash("voc2012-seg-test2"),
        skip_loading=False,
        task_type="segmentation" 
    )
    return exp

