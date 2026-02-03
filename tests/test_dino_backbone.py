import os
import sys
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from shared.model.modules import KeyEncoder_DINOv2_v6


def _run_case(name, dino_cfg):
    torch.manual_seed(0)
    model = KeyEncoder_DINOv2_v6(dino_cfg)
    model.train()

    x = torch.randn(2, 3, 112, 112, requires_grad=True)
    f16, f8, f4 = model(x)

    assert f16.shape[-2:] == (7, 7), f"{name}: f16 shape mismatch {f16.shape}"
    assert f8.shape[-2:] == (14, 14), f"{name}: f8 shape mismatch {f8.shape}"
    assert f4.shape[-2:] == (28, 28), f"{name}: f4 shape mismatch {f4.shape}"

    loss = f16.mean() + f8.mean() + f4.mean()
    loss.backward()


def test_dinov2_stub_interpolate():
    dino_cfg = {
        "source": "stub",
        "embedding_size": 384,
        "patch_size": 14,
        "num_prefix_tokens": 1,
        "layer_indices": [8, 9, 10, 11],
        "align": "interpolate",
        "target_stride": 16,
        "freeze": True,
        "resnet_pretrained": False,
    }
    _run_case("dinov2_stub", dino_cfg)


def test_dinov3_stub_learned():
    dino_cfg = {
        "source": "stub",
        "embedding_size": 384,
        "patch_size": 16,
        "num_prefix_tokens": 5,
        "layer_indices": [8, 9, 10, 11],
        "align": "learned",
        "target_stride": 16,
        "freeze": True,
        "resnet_pretrained": False,
    }
    _run_case("dinov3_stub", dino_cfg)


if __name__ == "__main__":
    test_dinov2_stub_interpolate()
    test_dinov3_stub_learned()
    print("OK")
