#!/usr/bin/env python3
"""Validate Blackwell sm_120, PyTorch, and ComfyUI container integrity."""

import asyncio
import importlib.util
import os
import sys


def validate_pytorch_and_blackwell():
    print("=== PyTorch & CUDA Blackwell Validation ===")
    import torch
    import torchvision
    import torchvision.ops
    import torchaudio

    print("PyTorch Version:   ", torch.__version__)
    print("CUDA Runtime:      ", torch.version.cuda)
    assert "12.8" in torch.version.cuda, f"FAIL: Expected CUDA 12.8, got {torch.version.cuda}"

    build_config = torch.__config__.show()
    assert "sm_120" in build_config, "FAIL: sm_120 compute capability not found in torch build"
    assert importlib.util.find_spec("xformers") is None, "FAIL: incompatible xformers package found"
    _ = torchvision.ops.nms
    print("✔ PyTorch Blackwell sm_120 and CUDA 12.8 verified successfully.")


def validate_base():
    validate_pytorch_and_blackwell()

    print("=== ComfyUI Core Registration Validation ===")
    sys.argv = ["main.py", "--cpu"]
    sys.path.append("/opt/ComfyUI")
    import comfy.options
    comfy.options.enable_args_parsing()
    import server
    import nodes

    print("Total registered nodes in clean ComfyUI core:", len(nodes.NODE_CLASS_MAPPINGS))
    assert len(nodes.NODE_CLASS_MAPPINGS) > 0, "FAIL: Zero nodes registered in ComfyUI core"
    print("✔ Clean ComfyUI core node registration healthy!")


def validate_custom_nodes():
    import numpy
    import numba
    import colour
    import insightface
    import onnx
    import mediapipe

    assert numpy.__version__.startswith("1."), f"FAIL: Expected NumPy 1.x, got {numpy.__version__}"
    print(f"NumPy version verified: {numpy.__version__}")

    sys.argv = ["main.py", "--cpu"]
    sys.path.append("/opt/ComfyUI")
    sys.path.append("/opt/ComfyUI/custom_nodes/ComfyUI-Manager/glob")
    import comfy.options
    comfy.options.enable_args_parsing()
    import server

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    s = server.PromptServer(loop)

    import execution
    import nodes
    nodes.init_extra_nodes()

    print("Total registered nodes in ComfyUI:", len(nodes.NODE_CLASS_MAPPINGS))
    assert "VHS_VideoCombine" in nodes.NODE_CLASS_MAPPINGS, "FAIL: VHS_VideoCombine missing"
    assert "VHS_LoadVideo" in nodes.NODE_CLASS_MAPPINGS, "FAIL: VHS_LoadVideo missing"
    assert "DownloadAndLoadLivePortraitModels" in nodes.NODE_CLASS_MAPPINGS, "FAIL: LivePortrait missing"
    assert "LatentSyncNode" in nodes.NODE_CLASS_MAPPINGS, "FAIL: LatentSync missing"
    assert any("AnimateDiff" in k for k in nodes.NODE_CLASS_MAPPINGS), "FAIL: AnimateDiff missing"
    assert any("GGUF" in k for k in nodes.NODE_CLASS_MAPPINGS), "FAIL: ComfyUI-GGUF nodes missing"
    print("✔ All production custom nodes registered successfully!")


def main():
    target = os.environ.get("TARGET")
    if not target and len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
        target = sys.argv[1]
    if not target:
        target = "nodes"

    print(f"Running automated container integrity validation for target: {target}")

    # Ensure sys.argv is clean for any subsequent argparse calls
    sys.argv = ["main.py", "--cpu"]

    if target == "base":
        validate_base()
    elif target in ("nodes", "full"):
        validate_pytorch_and_blackwell()
        validate_custom_nodes()
    else:
        print(f"Unknown target: {target}")
        sys.exit(1)


if __name__ == "__main__":
    main()
