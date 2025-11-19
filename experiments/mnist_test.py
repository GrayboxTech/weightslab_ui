"""
Minimal FashionCNN + WatcherEditor integration test.

- FashionCNN: pure PyTorch nn.Module
- Tests:
    * raw forward (bare model)
    * WatcherEditor init (ONNX export + deps)
    * forward through wrapped model
    * optional: neuron operation via model_op_neurons + forward
"""

import torch as th
import torch.nn as nn
import torch.nn.functional as F

from weightslab.backend.watcher_editor import WatcherEditor
from weightslab.utils.tools import model_op_neurons  # same helper you used with YOLO


# ---------------------- MODEL DEFINITION ---------------------- #

class FashionCNN(nn.Module):
    """
    Simple CNN for 28x28 grayscale images (e.g. MNIST/FashionMNIST).

    This is a *pure* PyTorch model; no NetworkWithOps, no custom WithNeuronOps
    modules. WatcherEditor + monkey_patch will later inject the neuron-wise
    machinery.
    """
    def __init__(self):
        super().__init__()

        # Input: [B, 1, 28, 28]
        self.layer1 = nn.Conv2d(in_channels=1, out_channels=4,
                                kernel_size=3, padding=1)  # -> [B, 4, 28, 28]
        self.bnorm1 = nn.BatchNorm2d(4)
        self.mpool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # -> [B, 4, 14, 14]

        self.layer2 = nn.Conv2d(in_channels=4, out_channels=4,
                                kernel_size=3)              # -> [B, 4, 12, 12]
        self.bnorm2 = nn.BatchNorm2d(4)
        self.mpool2 = nn.MaxPool2d(2)                       # -> [B, 4, 6, 6]

        self.fc = nn.Linear(in_features=4 * 6 * 6, out_features=10)  # -> [B, 10]

    def forward(self, x: th.Tensor) -> th.Tensor:
        # Block 1
        x = self.layer1(x)
        x = self.bnorm1(x)
        x = F.relu(x)
        x = self.mpool1(x)

        # Block 2
        x = self.layer2(x)
        x = self.bnorm2(x)
        x = F.relu(x)
        x = self.mpool2(x)

        # Flatten and classify
        x = x.view(x.size(0), -1)
        logits = self.fc(x)
        return logits


# ---------------------- TESTS ---------------------- #

def test_fashioncnn_raw():
    """
    Sanity check: plain FashionCNN forward on random input.
    """
    print("\n[TEST] FashionCNN raw forward")

    device = th.device("cpu")
    model = FashionCNN().to(device)

    dummy = th.randn(8, 1, 28, 28, device=device)

    try:
        with th.no_grad():
            out = model(dummy)
        print("  ✅ Raw forward OK")
        print("     logits shape:", out.shape)
    except Exception as e:
        print("  ❌ Raw forward FAILED")
        raise e


def test_fashioncnn_with_watcher():
    """
    Check FashionCNN + WatcherEditor:
    - raw forward
    - WatcherEditor init (ONNX export + deps)
    - wrapped forward
    - optional: model_op_neurons + forward
    """
    print("\n[TEST] FashionCNN + WatcherEditor")

    device = th.device("cpu")

    # 1. Build model and dummy input
    base_model = FashionCNN().to(device)
    dummy = th.randn(8, 1, 28, 28, device=device)

    # 2. Raw forward
    try:
        with th.no_grad():
            out = base_model(dummy)
        print("  ✅ Raw forward OK")
        print("     logits shape:", out.shape)
    except Exception:
        print("  ❌ Raw forward FAILED")
        raise

    # 3. Wrap with WatcherEditor (this will export to ONNX and build deps)
    try:
        wrapped = WatcherEditor(
            base_model,
            dummy_input=dummy,    # so WatcherEditor doesn't need model.input_shape
            device=device,
            print_graph=False,    # set True and add filename if you want a graph viz
        )
        print("  ✅ WatcherEditor init OK")
    except Exception:
        print("  ❌ WatcherEditor init FAILED")
        raise

    def module_name(root, module):
        for name, m in root.named_modules():
            if m is module:
                return name
        return f"<?{module.__class__.__name__}>"


    print("\n=== Dependencies in WatcherEditor (ONNX) ===")
    for (src, dst, dep_type) in wrapped.dependencies_with_ops:
        print(f"  {dep_type.name:8} {module_name(wrapped.model, src):15} -> {module_name(wrapped.model, dst):15}")


    # 4. Forward through wrapped model
    try:
        with th.no_grad():
            out_wrapped = wrapped(dummy)
        print("  ✅ Wrapped forward OK")
        print("     wrapped logits shape:", out_wrapped.shape)
    except Exception:
        print("  ❌ Wrapped forward FAILED")
        raise

    # 5. Optional: apply neuron operation via model_op_neurons
    try:
        print("  ℹ️  Applying model_op_neurons(wrapped, op=1, rand=False)...")
        model_op_neurons(wrapped, op=2, rand=False)

        # IMPORTANT: if operate() replaces modules, re-apply monkey patching
        wrapped.monkey_patch_model()

        with th.no_grad():
            out_after = wrapped(dummy)
        print("  ✅ operate() + forward OK")
        print("     logits after ops shape:", out_after.shape)
    except Exception as e:
        print("  ⚠️ operate() or post-op forward FAILED")
        raise e


def main():
    test_fashioncnn_raw()
    test_fashioncnn_with_watcher()


if __name__ == "__main__":
    main()
