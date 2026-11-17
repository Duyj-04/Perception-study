"""Check the pinned official architecture on CPU with random parameters.

No pretrained weights, natural-image predictions, or accuracy claims are made.
The source root defaults to the ignored cache described in the source manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    note_dir = Path(__file__).resolve().parent
    workspace = note_dir.parents[2]
    manifest = json.loads((note_dir / "network_source_manifest.json").read_text(encoding="utf-8"))
    source_root = (args.source_root or workspace / manifest["source_cache"]).resolve()
    for entry in manifest["files"]:
        path = source_root / entry["path"]
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            raise RuntimeError(f"Pinned source missing or changed: {path}")

    # These settings affect this process only; use the author's Torch fallback.
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(source_root))
    import cv2
    import numpy as np
    import torch
    import torch.nn.functional as F
    from depth_anything_v2 import dpt
    from depth_anything_v2.dinov2_layers import attention

    if Path(dpt.__file__).resolve() != source_root / "depth_anything_v2/dpt.py":
        raise RuntimeError("Imported a different Depth Anything installation")
    attention.XFORMERS_AVAILABLE = False
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    torch.manual_seed(0)
    model = dpt.DepthAnythingV2(
        encoder="vits", features=64, out_channels=[48, 96, 192, 384], use_clstoken=False
    ).cpu().eval()

    def shape_tree(value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            return list(value.shape)
        if isinstance(value, (tuple, list)):
            return [shape_tree(item) for item in value]
        return value

    events: list[dict[str, Any]] = []
    capture = False

    def make_hook(name: str):
        def hook(_module: Any, inputs: Any, output: Any) -> None:
            if capture:
                events.append({"module": name, "input": shape_tree(inputs), "output": shape_tree(output)})
        return hook

    def head_pre_hook(_module: Any, inputs: Any) -> None:
        if capture:
            events.append({"module": "selected_features", "output": shape_tree(inputs[0])})

    selected = {
        "pretrained.patch_embed.proj", "pretrained.patch_embed", "pretrained.norm",
        "pretrained.blocks.0.attn.qkv", "pretrained.blocks.0.mlp.fc1",
        "pretrained.blocks.0.mlp.fc2", "depth_head", "depth_head.scratch.output_conv1",
        "depth_head.scratch.output_conv2",
    }
    selected.update(f"pretrained.blocks.{i}" for i in range(12))
    for i in range(4):
        selected.update({f"depth_head.projects.{i}", f"depth_head.resize_layers.{i}"})
        selected.update({f"depth_head.scratch.layer{i+1}_rn", f"depth_head.scratch.refinenet{i+1}"})
    handles = [module.register_forward_hook(make_hook(name)) for name, module in model.named_modules() if name in selected]
    handles.append(model.depth_head.register_forward_pre_hook(head_pre_hook))

    cases: list[dict[str, Any]] = []
    for name, original_hw, expected_hw in [
        ("square_518", (518, 518), (518, 518)),
        ("rectangular_480x640", (480, 640), (518, 686)),
    ]:
        raw = np.zeros((*original_hw, 3), dtype=np.uint8)
        raw[:, :, 1] = 128
        raw[:, :, 2] = 255  # BGR=(0,128,255), used to check channel conversion.
        tensor, recovered_hw = model.image2tensor(raw, input_size=518)
        assert tensor.device.type == "cpu"
        assert tuple(recovered_hw) == original_hw
        assert tuple(tensor.shape) == (1, 3, *expected_hw)
        expected_rgb = torch.tensor([(1-.485)/.229, (128/255-.456)/.224, (0-.406)/.225])
        assert torch.allclose(tensor[0, :, 0, 0], expected_rgb, atol=1e-5)
        assert tensor.dtype == torch.float32

        events = []
        capture = True
        with torch.inference_mode():
            forward = model(tensor)
        capture = False
        ph, pw = expected_hw[0] // 14, expected_hw[1] // 14
        by_name = {event["module"]: event["output"] for event in events}
        assert by_name["pretrained.patch_embed"] == [1, ph*pw, 384]
        assert by_name["pretrained.blocks.0"] == [1, ph*pw+1, 384]
        assert by_name["pretrained.blocks.0.attn.qkv"] == [1, ph*pw+1, 1152]
        assert by_name["selected_features"] == [[[1, ph*pw, 384], [1, 384]]] * 4
        for i, (channels, height, width) in enumerate([
            (48, ph*4, pw*4), (96, ph*2, pw*2), (192, ph, pw),
            (384, (ph+1)//2, (pw+1)//2),
        ]):
            assert by_name[f"depth_head.resize_layers.{i}"] == [1, channels, height, width]
        for level, height, width in [(4, ph, pw), (3, ph*2, pw*2), (2, ph*4, pw*4), (1, ph*8, pw*8)]:
            assert by_name[f"depth_head.scratch.refinenet{level}"] == [1, 64, height, width]
        assert by_name["depth_head"] == [1, 1, *expected_hw]
        assert list(forward.shape) == [1, *expected_hw]

        with torch.inference_mode():
            native = model.infer_image(raw, input_size=518)
            restored = F.interpolate(forward[:, None], original_hw, mode="bilinear", align_corners=True)[0, 0].numpy()
        assert native.shape == original_hw
        assert native.dtype == np.float32 and np.isfinite(native).all()
        assert (native >= 0).all()
        assert np.allclose(native, restored, rtol=1e-5, atol=1e-6)
        cases.append({
            "name": name, "original_hwc": list(raw.shape), "preprocessed_bchw": list(tensor.shape),
            "patch_grid": [ph, pw], "selected_layer_indices_zero_based": [2, 5, 8, 11],
            "events": events, "forward_output": list(forward.shape),
            "infer_image_output": list(native.shape), "preprocessing_values_checked": True,
            "native_wrapper_matches_restored_forward": True, "status": "pass",
        })
        print(f"PASS {name}: {tuple(tensor.shape)} -> {tuple(forward.shape)} -> {native.shape}", flush=True)

    with torch.inference_mode():
        batched = model(torch.zeros(2, 3, 56, 84))
    assert tuple(batched.shape) == (2, 56, 84)
    rejected = False
    try:
        with torch.inference_mode():
            model(torch.zeros(1, 3, 55, 84))
    except AssertionError as error:
        rejected = "multiple of patch height" in str(error)
    assert rejected, "Non-multiple input must be rejected by the pinned PatchEmbed"
    for handle in handles:
        handle.remove()

    result = {
        "validation_kind": "architecture_shapes_random_parameters",
        "source_commit": manifest["commit"], "pretrained_weights_loaded": False,
        "environment": {"python": platform.python_version(), "executable": sys.executable,
                        "torch": torch.__version__, "numpy": np.__version__, "opencv": cv2.__version__,
                        "device": "cpu", "torch_threads": torch.get_num_threads(), "attention_backend": "torch_fallback"},
        "model_config": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384], "use_clstoken": False},
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "cases": cases, "batch_case": {"input": [2, 3, 56, 84], "output": list(batched.shape), "status": "pass"},
        "non_multiple_height_rejected": rejected, "status": "pass",
    }
    output = args.output or note_dir / "network_shape_trace.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Recorded {len(cases)} traced cases, batch and invalid-input checks: {output}")


if __name__ == "__main__":
    main()
