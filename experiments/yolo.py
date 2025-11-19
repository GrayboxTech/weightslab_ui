"""
YOLOv5 + WeightsLab WatcherEditor integration test.

- Loads YOLOv5 via torch.hub (Ultralytics repo)
- Wraps the internal model in an Fx-friendly nn.Module
- Checks:
    * raw forward
    * WatcherEditor init
    * wrapped forward
"""

import traceback

import torch
import torch.nn as nn

from weightslab.backend.watcher_editor import WatcherEditor


class FxFriendlyYOLOv5(nn.Module):
    """
    Wrapper around a YOLOv5 model loaded from torch.hub.

    Exposes:
      - self.input_shape  (for WatcherEditor dummy_input logic)
      - forward(x) that returns a single Tensor with predictions
    """
    def __init__(
        self,
        weights: str = "yolov5s",        # 'yolov5n', 'yolov5s', 'yolov5m', ...
        img_size=(640, 640),
        device: str = "cpu",
        hub_repo: str = "ultralytics/yolov5",
        hub_source: str | None = None,  # e.g. "local" if you have the repo
        pretrained: bool = True,
    ):
        super().__init__()

        self.device = device
        self.input_shape = (1, 3, img_size[0], img_size[1])

        # --- 1. Load YOLOv5 from torch.hub ---
        hub_kwargs = {}
        if hub_source is not None:
            # e.g. hub_source="local" if you've cloned ultralytics/yolov5
            hub_kwargs["source"] = hub_source

        # This returns a hub model wrapper (not just nn.Module)
        # Usually a custom Ultralytics object with .model attribute inside.
        self.hub_model = torch.hub.load(
            hub_repo,
            weights,
            pretrained=pretrained,
            **hub_kwargs,
        ).to(device)

        # Ensure eval mode for inference/tracing
        self.hub_model.eval()

        # Try to get the core nn.Module (no preprocessing / NMS)
        # Many YOLOv5 hub versions expose `.model` as the underlying Model.
        self.core = getattr(self.hub_model, "model", self.hub_model)

    def forward(self, x: torch.Tensor):
        """
        Forward pass through the core YOLOv5 network, without any pre/post-processing.

        We try:
          - core._forward_once(x)
          - core.forward_once(x)
          - core(x)

        and always return the "predictions" tensor (first element if it's a list/tuple).
        """
        core = self.core

        if hasattr(core, "_forward_once"):
            out = core._forward_once(x)
        elif hasattr(core, "forward_once"):
            out = core.forward_once(x)
        else:
            out = core(x)

        # Training mode may return (preds, aux), so keep only preds
        if isinstance(out, (list, tuple)):
            out = out[0]

        return out


# ---------------------- TESTS ---------------------- #

def test_yolov5_raw():
    """
    Sanity check: load YOLOv5, run a raw forward pass on random input.
    """
    print("\n[TEST] Raw FxFriendlyYOLOv5 forward")

    device = "cpu"  # change to "cuda" if available and desired

    model = FxFriendlyYOLOv5(
        weights="yolov5s",
        img_size=(640, 640),
        device=device,
    ).to(device)

    dummy = torch.randn(model.input_shape, device=device)

    try:
        with torch.no_grad():
            out = model(dummy)

        print("  ✅ Raw forward OK")
        print("     type:", type(out))
        print("     shape:", getattr(out, "shape", None))
    except Exception:
        print("  ❌ Raw forward FAILED")
        traceback.print_exc()


def test_yolov5_with_watcher():
    """
    Check if FxFriendlyYOLOv5 can be wrapped with WatcherEditor:
    - raw forward
    - WatcherEditor init
    - wrapped forward
    - optional operate()
    """
    print("\n[TEST] FxFriendlyYOLOv5 + WatcherEditor")

    device = "cpu"

    # 1. Build model and dummy input
    model = FxFriendlyYOLOv5(
        weights="yolov5s",
        img_size=(640, 640),
        device=device,
    ).to(device)

    dummy = torch.randn(model.input_shape, device=device)

    # 2. Raw forward
    try:
        with torch.no_grad():
            out = model(dummy)
        print("  ✅ Raw forward OK")
        print("     out shape:", getattr(out, "shape", None))
    except Exception:
        print("  ❌ Raw forward FAILED")
        traceback.print_exc()
        return

    # 3. Wrap with WatcherEditor
    try:
        wrapped = WatcherEditor(
            model,
            dummy_input=dummy,
            device=device,
            print_graph=False,
        )
        print("  ✅ WatcherEditor init OK")
    except Exception:
        print("  ❌ WatcherEditor init FAILED")
        traceback.print_exc()
        return

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
        with torch.no_grad():
            out_wrapped = wrapped(dummy)
        print("  ✅ Wrapped forward OK")
        print("     wrapped out shape:", getattr(out_wrapped, "shape", None))
    except Exception:
        print("  ❌ Wrapped forward FAILED")
        traceback.print_exc()
        return

    # 5. Optional: try an operation if WatcherEditor exposes it
    if hasattr(wrapped, "operate"):
        try:
            print("  ℹ️  Trying wrapped.operate(0, 1, 1)...")
            wrapped.operate(0, 1, 1)
            with torch.no_grad():
                _ = wrapped(dummy)
            print("  ✅ operate() + forward OK")
        except Exception:
            print("  ⚠️ operate() failed (could be layer mapping / deps)")
            traceback.print_exc()


def main():
    test_yolov5_raw()
    test_yolov5_with_watcher()


if __name__ == "__main__":
    main()
