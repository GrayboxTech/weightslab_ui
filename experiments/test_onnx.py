"""
Extra models to stress-test WatcherEditor + ONNX deps.

We want to see:
- SAME vs INCOMING deps (via channel changes)
- REC deps (via residual Add & Concat)
"""

import torch as th
import torch.nn as nn
import torch.nn.functional as F

from weightslab.backend.watcher_editor import WatcherEditor
from weightslab.utils.tools import model_op_neurons

import inspect
print("[DEBUG] WatcherEditor source file:", inspect.getsourcefile(WatcherEditor))


# ---------------------- UTIL ---------------------- #

def module_name(root: nn.Module, module: nn.Module) -> str:
    """Resolve a module object back to its dotted name under root."""
    for name, m in root.named_modules():
        if m is module:
            return name
    return f"<?{module.__class__.__name__}>"


def print_dependencies(tag: str, wrapped: WatcherEditor):
    print(f"\n=== Dependencies in WatcherEditor (ONNX) :: {tag} ===")
    for (src, dst, dep_type) in wrapped.dependencies_with_ops:
        print(f"  {dep_type.name:8} {module_name(wrapped.model, src):20} -> {module_name(wrapped.model, dst):20}")


# ==================================================
# 1) ResidualCNN  (tests Add → REC deps)
# ==================================================

class ResidualCNN(nn.Module):
    """
    Simple residual CNN with one skip connection:
      x -> conv1 -> bn1 -> relu -> conv2 -> bn2 -> relu
                                ↘------------(+)----> ...
    """
    def __init__(self):
        super().__init__()

        # Input: [B, 3, 32, 32]
        self.conv1 = nn.Conv2d(3, 4, kernel_size=3, padding=1)   # [B, 4, 32, 32]
        self.bn1   = nn.BatchNorm2d(4)

        # Projection for residual to match conv2/bn2 channels (4 -> 8)
        self.proj  = nn.Conv2d(4, 8, kernel_size=1)              # [B, 8, 32, 32]

        self.conv2 = nn.Conv2d(4, 8, kernel_size=3, padding=1)   # [B, 8, 32, 32]
        self.bn2   = nn.BatchNorm2d(8)

        self.pool  = nn.MaxPool2d(2)                             # [B, 8, 16, 16]
        self.fc    = nn.Linear(8 * 16 * 16, 10)

    def forward(self, x: th.Tensor) -> th.Tensor:
        # First block
        x = self.conv1(x)    # [B, 4, 32, 32]
        x = self.bn1(x)
        x = F.relu(x)

        residual = x         # [B, 4, 32, 32]

        # Second block
        x = self.conv2(x)    # [B, 8, 32, 32]
        x = self.bn2(x)
        x = F.relu(x)

        # Project residual to 8 channels before adding
        res_proj = self.proj(residual)   # [B, 8, 32, 32]
        x = x + res_proj                 # [B, 8, 32, 32]

        x = self.pool(x)                 # [B, 8, 16, 16]
        x = x.view(x.size(0), -1)
        logits = self.fc(x)
        return logits



def test_residualcnn_with_watcher():
    print("\n[TEST] ResidualCNN + WatcherEditor")
    device = th.device("cpu")
    base_model = ResidualCNN().to(device)
    dummy = th.randn(4, 3, 32, 32, device=device)

    # Raw forward
    with th.no_grad():
        out = base_model(dummy)
    print("  ✅ Raw forward OK (ResidualCNN)  logits:", out.shape)

    # Wrap
    wrapped = WatcherEditor(
        base_model,
        dummy_input=dummy,
        device=device,
        print_graph=False,
    )
    print("  ✅ WatcherEditor init OK (ResidualCNN)")
    print(wrapped.dependencies_with_ops)

    print_dependencies("ResidualCNN", wrapped)

    # Forward through wrapped
    with th.no_grad():
        out_w = wrapped(dummy)
    print("  ✅ Wrapped forward OK (ResidualCNN)  logits:", out_w.shape)

    # Optional neuron operations
    try:
        print("  ℹ️  Applying model_op_neurons(wrapped, op=2, rand=False)...")
        model_op_neurons(wrapped, op=2, rand=False)
        wrapped.monkey_patch_model()
        with th.no_grad():
            out_after = wrapped(dummy)
        print("  ✅ operate() + forward OK (ResidualCNN)  logits:", out_after.shape)
    except Exception as e:
        print("  ⚠️ operate() failed on ResidualCNN:", e)
        raise


# ==================================================
# 2) ConcatBranchCNN  (tests Concat → REC deps + bypass)
# ==================================================

class ConcatBranchCNN(nn.Module):
    """
    Two branches from a common stem, merged by concat along channel dim:

        stem -> b1 -> out1 --\
                              +--> concat -> conv_out -> ...
        stem -> b2 -> out2 --/

    ONNX should emit a Concat node → REC deps between branch producers.
    """
    def __init__(self):
        super().__init__()

        # Input: [B, 3, 32, 32]
        self.stem = nn.Conv2d(3, 4, kernel_size=3, padding=1)    # [B, 4, 32, 32]

        self.branch1_conv = nn.Conv2d(4, 4, kernel_size=3, padding=1)
        self.branch1_bn   = nn.BatchNorm2d(4)

        self.branch2_conv = nn.Conv2d(4, 4, kernel_size=3, padding=1)
        self.branch2_bn   = nn.BatchNorm2d(4)

        # After concat: channels = 4 + 4 = 8
        self.conv_out = nn.Conv2d(8, 8, kernel_size=3, padding=1)
        self.bn_out   = nn.BatchNorm2d(8)
        self.pool     = nn.MaxPool2d(2)                          # [B, 8, 16, 16]
        self.fc       = nn.Linear(8 * 16 * 16, 5)

    def forward(self, x: th.Tensor) -> th.Tensor:
        x = self.stem(x)
        x = F.relu(x)

        b1 = self.branch1_conv(x)
        b1 = self.branch1_bn(b1)
        b1 = F.relu(b1)

        b2 = self.branch2_conv(x)
        b2 = self.branch2_bn(b2)
        b2 = F.relu(b2)

        # Concat along channel dim = 1
        x = th.cat([b1, b2], dim=1)

        x = self.conv_out(x)
        x = self.bn_out(x)
        x = F.relu(x)

        x = self.pool(x)
        x = x.view(x.size(0), -1)
        logits = self.fc(x)
        return logits


def test_concatcnn_with_watcher():
    print("\n[TEST] ConcatBranchCNN + WatcherEditor")
    device = th.device("cpu")
    base_model = ConcatBranchCNN().to(device)
    dummy = th.randn(4, 3, 32, 32, device=device)

    # Raw forward
    with th.no_grad():
        out = base_model(dummy)
    print("  ✅ Raw forward OK (ConcatBranchCNN)  logits:", out.shape)

    # Wrap
    wrapped = WatcherEditor(
        base_model,
        dummy_input=dummy,
        device=device,
        print_graph=False,
    )
    print("  ✅ WatcherEditor init OK (ConcatBranchCNN)")

    print_dependencies("ConcatBranchCNN", wrapped)

    # Forward through wrapped
    with th.no_grad():
        out_w = wrapped(dummy)
    print("  ✅ Wrapped forward OK (ConcatBranchCNN)  logits:", out_w.shape)

    # Optional neuron operations
    try:
        print("  ℹ️  Applying model_op_neurons(wrapped, op=2, rand=False)...")
        model_op_neurons(wrapped, op=2, rand=False)
        wrapped.monkey_patch_model()
        with th.no_grad():
            out_after = wrapped(dummy)
        print("  ✅ operate() + forward OK (ConcatBranchCNN)  logits:", out_after.shape)
    except Exception as e:
        print("  ⚠️ operate() failed on ConcatBranchCNN:", e)
        raise


# ==================================================
# 3) ChannelChangeCNN  (tests INCOMING deps)
# ==================================================

class ChannelChangeCNN(nn.Module):
    """
    Sequential convs with changing channel counts to trigger INCOMING deps
    (src C != dst C).
    """
    def __init__(self):
        super().__init__()

        # Input: [B, 3, 32, 32]
        self.conv1 = nn.Conv2d(3, 4, kernel_size=3, padding=1)   # [B, 4, 32, 32]
        self.bn1   = nn.BatchNorm2d(4)

        self.conv2 = nn.Conv2d(4, 7, kernel_size=3, padding=1)   # [B, 7, 32, 32]
        self.bn2   = nn.BatchNorm2d(7)

        self.conv3 = nn.Conv2d(7, 10, kernel_size=3, padding=1) # [B, 10, 32, 32]
        self.bn3   = nn.BatchNorm2d(10)

        self.pool  = nn.AdaptiveAvgPool2d((1, 1))                # [B, 10, 1, 1]
        self.fc    = nn.Linear(10, 3)

    def forward(self, x: th.Tensor) -> th.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))   # C=4
        x = F.relu(self.bn2(self.conv2(x)))   # C=7
        x = F.relu(self.bn3(self.conv3(x)))   # C=10
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        logits = self.fc(x)
        return logits


def test_channelchangecnn_with_watcher():
    print("\n[TEST] ChannelChangeCNN + WatcherEditor")
    device = th.device("cpu")
    base_model = ChannelChangeCNN().to(device)
    dummy = th.randn(4, 3, 32, 32, device=device)

    # Raw forward
    with th.no_grad():
        out = base_model(dummy)
    print("  ✅ Raw forward OK (ChannelChangeCNN)  logits:", out.shape)

    # Wrap
    wrapped = WatcherEditor(
        base_model,
        dummy_input=dummy,
        device=device,
        print_graph=False,
    )
    print("  ✅ WatcherEditor init OK (ChannelChangeCNN)")

    print_dependencies("ChannelChangeCNN", wrapped)

    # Forward through wrapped
    with th.no_grad():
        out_w = wrapped(dummy)
    print("  ✅ Wrapped forward OK (ChannelChangeCNN)  logits:", out_w.shape)

    # Optional neuron operations
    try:
        print("  ℹ️  Applying model_op_neurons(wrapped, op=2, rand=False)...")
        model_op_neurons(wrapped, op=2, rand=False)
        wrapped.monkey_patch_model()
        with th.no_grad():
            out_after = wrapped(dummy)
        print("  ✅ operate() + forward OK (ChannelChangeCNN)  logits:", out_after.shape)
    except Exception as e:
        print("  ⚠️ operate() failed on ChannelChangeCNN:", e)
        raise


# ==================================================
# MAIN
# ==================================================

def main():
    test_concatcnn_with_watcher()
    test_channelchangecnn_with_watcher()
    test_residualcnn_with_watcher()


if __name__ == "__main__":
    main()
