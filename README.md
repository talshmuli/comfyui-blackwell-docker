# ComfyUI & Blender VFX Studio Bundle for NVIDIA Blackwell (RTX 5090 / 5080 - sm_120)

[![Build & Publish ComfyUI Blackwell Bundle](https://github.com/talshmuli/comfyui-blackwell-bundle/actions/workflows/build.yml/badge.svg)](https://github.com/talshmuli/comfyui-blackwell-bundle/actions/workflows/build.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![CUDA: 12.8](https://img.shields.io/badge/CUDA-12.8-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![Compute: sm_120](https://img.shields.io/badge/Architecture-Blackwell%20sm__120-orange.svg)](https://www.nvidia.com/en-us/geforce/rtx-5090/)

A production-grade, multi-stage containerized generative VFX studio built specifically for **NVIDIA GeForce RTX 5090 and RTX 5080** GPUs utilizing the **Blackwell architecture (`sm_120`)**, **CUDA 12.8+**, and **Blender 4.4 LTS**.

---

## 1. Executive Summary & One-Stop-Shop Studio Motivation

### The Problem: Blackwell Kernel Mismatch & Fragmented Generative VFX
NVIDIA GeForce RTX 50-series GPUs (RTX 5090 / 5080) are engineered on the cutting-edge **Blackwell** microarchitecture, requiring CUDA compute capability **`sm_120`**.

Standard ComfyUI Docker containers ship with stable PyTorch distributions (PyTorch 2.4/2.5 on CUDA 12.1/12.4), which only provide precompiled CUDA kernels up to `sm_90` (Hopper/Ada Lovelace). Running state-of-the-art generative video diffusion on an RTX 5090 with standard images triggers the fatal CUDA error:
```text
RuntimeError: CUDA error: no kernel image is available for execution on the device
```

Furthermore, advanced cinematic pipelines—such as stylized 3D animation, facial performance capture, and audio-driven lip-sync—require a synchronized stack uniting:
1. Deep learning frameworks compiled against native Blackwell `sm_120` CUDA 12.8 kernels.
2. ComfyUI with a web of delicate video custom nodes (LivePortrait, LatentSync, VideoHelperSuite, AnimateDiff).
3. 3D kinematic modeling engines (Blender 4.4 LTS) for procedural rigging, facial blendshapes, and geometric AOV passes (Depth, Normal, OpenPose).

### The Solution: 4-Tier Matrix — From Lean Microservices to Full Monolith
This repository offers both deployment strategies, so you choose the right image for each workload:
* **Lean microservices** (Recommended): `nodes` + `blender-worker` as separate services share a Docker volume and talk over an isolated bridge. Blender crashes can never kill the ComfyUI web UI.
* **All-in-one monolith** (`full`): Single container with ComfyUI, all custom nodes *and* Blender 4.4 LTS pre-installed — ideal for single-GPU workstations where simplicity matters more than isolation.

This repository solves these issues through a **4-Tier Multi-Stage Image Matrix** and **Modular Docker Compose Orchestration**:
* **ComfyUI Nodes Worker (~16 GB):** Runs the web interface and neural diffusion pipeline.
* **Blender VFX Worker (~3.5 GB):** Lean, standalone headless Blender 4.4 LTS container with OptiX/CUDA acceleration.
* **Full Studio Monolith (~19 GB):** All-in-one container with ComfyUI, all custom nodes, and Blender 4.4 LTS — for single-service deployments.
* **Shared Render Bus (`passes_bus`):** High-speed inter-container volume bridging Blender render outputs directly into ComfyUI's input pipeline (`/workspace/passes` $\leftrightarrow$ `/opt/ComfyUI/input/passes`).
* **Isolated Studio Network (`vfx-net`):** High-speed internal Docker bridge for API-driven automation.

```
       ┌────────────────────────────────────────────────────────┐
       │                   Docker Compose VFX                   │
       │                                                        │
       │   ┌───────────────────┐        ┌───────────────────┐   │
       │   │  comfyui-nodes    │        │  blender-worker   │   │
       │   │  (CUDA 12.8 /     │        │  (Blender 4.4.3   │   │
       │   │   sm_120 / Nodes) │        │   OptiX / CUDA)   │   │
       │   └─────────▲─────────┘        └─────────┬─────────┘   │
       │             │                            │             │
       │             │   passes_bus (Shared Vol)  │             │
       │             └─── /opt/ComfyUI/input/ ────┘             │
       │                  passes <──> /workspace/passes         │
       │                                                        │
       │   ═════════════════ vfx-net (Bridge) ═══════════════   │
       └────────────────────────────────────────────────────────┘
```

---

## 2. Who This Bundle Is For (and What It Is NOT)

### 🎯 Who Will Enjoy This Bundle
* **NVIDIA Blackwell GPU Owners (RTX 5090 / 5080 / 5070):** Creators and engineers seeking native hardware `sm_120` execution with zero CUDA kernel mismatches or manual compilation headaches.
* **Generative Video & Digital Human Artists:** Anyone building facial performance reenactment, temporal diffusion, and audio-driven lip-sync pipelines (`LivePortrait`, `LatentSync`, `VideoHelperSuite`, `AnimateDiff-Evolved`).
* **3D Animation & Technical VFX Studios:** Artists integrating 3D spatial kinematics from Blender 4.4 LTS (depth passes, normals, OpenPose AOVs) directly into ComfyUI neural pipelines.
* **Quantized Model Practitioners:** Workstations running heavy 12B+ diffusion models (FLUX.1 Dev/Schnell, SD 3.5, Wan 2.1, HunyuanVideo) at 4-bit, 5-bit, or 8-bit precision using native `ComfyUI-GGUF` and FP8.
* **Security-Conscious Studios & Homelabs:** Teams requiring verified container security (0 Critical CVEs, Pillow 13 source compilation, automated Aqua Trivy audits, and locked PyTorch kernels via `PIP_CONSTRAINT`).

### 🚫 What This Bundle Is NOT For
* **Non-NVIDIA or Legacy Hardware:** Not intended for AMD GPUs (ROCm), Apple Silicon (MPS), Intel GPUs (ARC), or older pre-Turing NVIDIA architectures (GTX 10-series or earlier).
* **Experimental / Fragile Third-Party C++ Formats:** Not for experimental proprietary engines (such as MIT Nunchaku SVDQuant) that require hardcoded C++ wheels tied to specific PyTorch builds or force lowering container security (`security_level = weak`). We prioritize industry-standard, rock-solid **Native FP8 and GGUF**.
* **Global Experimental Attention Kernel Overrides:** We intentionally avoid forcing global experimental attention kernels (like `--use-sage-attention`) that are known to produce black frames on video models like Wan 2.1 and SD 1.5.
* **Barebones "Clean Slate" Minimalists:** If you only need an empty 50MB Python container and prefer manually downloading every node and debugging dependency conflicts from scratch, the pre-configured video/3D workstation may be heavier than what you need (though our `base` profile provides a clean starting point).

---

## 3. 4-Tier Multi-Stage Image Matrix

| Target Stage | Canonical Tag | Estimated Size | Included Components & Features | Primary Use Case |
| :--- | :--- | :--- | :--- | :--- |
| **`base`** | `base-cuda128-sm120` | ~12 GB | • Upstream `ai-dock/comfyui:latest-cuda`<br>• Official PyTorch Nightly CUDA 12.8 (`sm_120`)<br>• CUDA 12.8 driver runtime (`cudnn>=9.20`, `cublas>=12.8`)<br>• Clean ComfyUI core (0 custom nodes)<br>• Incompatible `xformers` uninstalled | Minimalist runtime, core testing, building custom node sets from scratch. |
| **`nodes`** *(Default)* | `nodes-cuda128-sm120`<br>`latest` | ~16 GB | • All `base` components<br>• `ComfyUI-VideoHelperSuite` (pinned `593fcb07`)<br>• `ComfyUI-LivePortraitKJ` (Facial dynamics)<br>• `ComfyUI-LatentSyncWrapper` (Audio lip-sync)<br>• `ComfyUI-AnimateDiff-Evolved` (pinned `b3e508a`)<br>• `ComfyUI-3D-Pack` (Mesh & UV processing)<br>• `ComfyUI-GGUF` (Quantized FLUX, SD3.5, Wan2.1 & Hunyuan)<br>• Pinned ML packages (`diffusers<0.31`, `insightface`, `mediapipe`, `gguf`) | Production generative video, character animation, facial performance capture, quantized FLUX/video diffusion. |
| **`blender-worker`** | `blender-worker`<br>`blender-4.4-worker` | ~3.5 GB | • Minimal `nvidia/cuda:12.8.0-base-ubuntu24.04`<br>• Official Blender 4.4.3 LTS Headless<br>• Native OptiX & CUDA Blackwell rendering<br>• `ComfyUI-BlenderAI-node` pre-installed<br>• Automated batch render runner (`blender_runner.py`) | Headless geometric pass generation (Depth, Normal, OpenPose), procedural 3D rigging. Pairs with `nodes` via shared volume. |
| **`full`** | `full-cuda128-sm120` | ~19 GB | • All `nodes` components (Video, Portrait, 3D, and GGUF)<br>• Official Blender 4.4.3 LTS Headless<br>• `ComfyUI-BlenderAI-node` addon<br>• Native OptiX & CUDA Blackwell rendering | All-in-one single-container studio — ComfyUI + all custom nodes + Blender 4.4 in one image. Ideal for single-GPU workstations. |

---

## 4. Quickstart & Docker Compose Orchestration

### Prerequisites
* NVIDIA Driver $\ge$ 570.xx installed on the host.
* NVIDIA Container Toolkit installed (`nvidia-smi` functional in Docker).
* Docker Compose v2.20+.

### Option A: Modular Studio Pipeline (Recommended)
Launches the synchronized microservices architecture: ComfyUI with all video nodes alongside the dedicated Blender headless worker.

```bash
docker compose --profile modular up -d
```
* **ComfyUI UI:** [`http://localhost:8188`](http://localhost:8188)
* **Shared Render Bus:** Any pass generated by Blender in `/workspace/passes` is immediately readable in ComfyUI at `/opt/ComfyUI/input/passes`.

### Option B: Developer / Clean Core Mode
Launches the clean, minimal base ComfyUI container alongside Blender:
```bash
docker compose --profile dev up -d
```

### Option C: Standalone ComfyUI (No Blender)
Runs only the ComfyUI nodes workstation:
```bash
docker compose --profile nodes up -d
```

### Option D: Standalone Base (Minimalist Core)
Runs only the clean core ComfyUI container:
```bash
docker compose --profile base up -d
```

### Option E: Full Studio Monolith (All-In-One)
Runs the single all-in-one container with ComfyUI + all nodes + Blender 4.4 LTS bundled together:
```bash
docker compose --profile monolith up -d
```
> **Note:** The `full` image bundles everything for single-GPU workstations where simplicity matters more than service isolation. For production studios with high availability requirements, the `modular` profile (Option A) is recommended.

### Custom Ports, Network Binding & Permissions
To customize host ports, restrict access to localhost, or match host user ownership:
```bash
# Localhost-only binding (enhanced privacy against local LAN/Wi-Fi snooping)
COMFYUI_BIND_IP=127.0.0.1 docker compose --profile modular up -d

# Custom port mapping
COMFYUI_PORT=8189 docker compose --profile modular up -d

# Custom user/group ownership for generated output files
PUID=1000 PGID=1000 docker compose --profile modular up -d
```
> **Remote Access Tip (Tailscale / VPN):** For secure remote generation from phones, laptops, or remote offices without port-forwarding, run Tailscale on your host machine (`tailscale up`). You can access ComfyUI directly at `http://<tailscale-ip>:8188` across your private encrypted WireGuard mesh.

---

## 5. Compose Configuration Specification

The complete production orchestration definition (`docker-compose.yml`):

```yaml
services:
  comfyui:
    image: ${COMFYUI_IMAGE:-ghcr.io/talshmuli/comfyui-blackwell-bundle:nodes-cuda128-sm120}
    container_name: comfyui-nodes
    profiles: ["modular", "nodes"]
    restart: unless-stopped
    shm_size: ${SHM_SIZE:-16gb}
    ports:
      - "${COMFYUI_BIND_IP:-0.0.0.0}:${COMFYUI_PORT:-8188}:8188"
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - NVIDIA_DRIVER_CAPABILITIES=all
      - WORKSPACE_UID=${PUID:-1000}
      - WORKSPACE_GID=${PGID:-1000}
      - UMASK=${UMASK:-0022}
      - PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
      - COMFYUI_PORT_LOCAL=${COMFYUI_PORT:-8188}
      - COMFYUI_ARGS=--listen 0.0.0.0 --reserve-vram ${RESERVE_VRAM:-1.5} ${COMFYUI_FLAGS:---preview-method auto --normalvram}
      - COMFYUI_FLAGS=--listen 0.0.0.0 --reserve-vram ${RESERVE_VRAM:-1.5} ${COMFYUI_FLAGS:---preview-method auto --normalvram}
      - WEB_ENABLE_AUTH=false
      - CF_QUICK_TUNNELS=false
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    volumes:
      - ./data/models:/opt/ComfyUI/models
      - ./data/output:/opt/ComfyUI/output
      - passes_bus:/opt/ComfyUI/input/passes
      # Optional: Mount additional community custom nodes individually to preserve pre-installed nodes:
      # - ./data/extra_custom_nodes/my-node:/opt/ComfyUI/custom_nodes/my-node
    networks:
      - vfx-net

  blender-worker:
    image: ${BLENDER_IMAGE:-ghcr.io/talshmuli/comfyui-blackwell-bundle:blender-worker}
    container_name: blender-worker
    profiles: ["modular", "dev"]
    restart: unless-stopped
    shm_size: ${SHM_SIZE:-16gb}
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - NVIDIA_DRIVER_CAPABILITIES=all
      - COMFYUI_API_URL=http://comfyui:8188
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    volumes:
      - ./data/blender_scenes:/workspace/scenes
      - passes_bus:/workspace/passes
    networks:
      - vfx-net

  comfyui-full:
    image: ${COMFYUI_FULL_IMAGE:-ghcr.io/talshmuli/comfyui-blackwell-bundle:full-cuda128-sm120}
    container_name: comfyui-full
    profiles: ["monolith", "full"]
    restart: unless-stopped
    shm_size: ${SHM_SIZE:-16gb}
    ports:
      - "${COMFYUI_BIND_IP:-0.0.0.0}:${COMFYUI_PORT:-8188}:8188"
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - NVIDIA_DRIVER_CAPABILITIES=all
      - WORKSPACE_UID=${PUID:-1000}
      - WORKSPACE_GID=${PGID:-1000}
      - UMASK=${UMASK:-0022}
      - PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
      - COMFYUI_PORT_LOCAL=${COMFYUI_PORT:-8188}
      - COMFYUI_ARGS=--listen 0.0.0.0 --reserve-vram ${RESERVE_VRAM:-1.5} ${COMFYUI_FLAGS:---preview-method auto --normalvram}
      - COMFYUI_FLAGS=--listen 0.0.0.0 --reserve-vram ${RESERVE_VRAM:-1.5} ${COMFYUI_FLAGS:---preview-method auto --normalvram}
      - WEB_ENABLE_AUTH=false
      - CF_QUICK_TUNNELS=false
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    volumes:
      - ./data/models:/opt/ComfyUI/models
      - ./data/output:/opt/ComfyUI/output
      - ./data/blender_scenes:/workspace/scenes
    networks:
      - vfx-net

volumes:
  passes_bus:

networks:
  vfx-net:
    driver: bridge
```

---

## 6. Strict Semantic Versioning & Immutability

To guarantee reproducibility in production studio environments, images are published with strict version tagging:

1. **Immutable Exact SemVer Tags (Never overwritten):**
   * `base`: `ghcr.io/talshmuli/comfyui-blackwell-bundle:v0.2.2-torch2.12.0.dev-cu128-base-YYYYMMDD-r<RUN_ID>`
   * `nodes`: `ghcr.io/talshmuli/comfyui-blackwell-bundle:v0.2.2-torch2.12.0.dev-cu128-nodes-YYYYMMDD-r<RUN_ID>`
   * `full`: `ghcr.io/talshmuli/comfyui-blackwell-bundle:v0.2.2-torch2.12.0.dev-cu128-full-YYYYMMDD-r<RUN_ID>`
   * `blender-worker`: `ghcr.io/talshmuli/comfyui-blackwell-bundle:blender4.4.3-cuda128-YYYYMMDD-r<RUN_ID>`
2. **Canonical Series Tags (Updated per stable release):**
   * `base-cuda128-sm120`
   * `nodes-cuda128-sm120` (also alias `v0.2.2-cuda128-sm120`)
   * `full-cuda128-sm120`
   * `blender-worker`
3. **Latest Tag:**
   * `ghcr.io/talshmuli/comfyui-blackwell-bundle:latest` points directly to the `nodes` production workstation.

---

## 7. Automated CI Pipeline & Verification

GitHub Actions workflow ([`.github/workflows/build.yml`](.github/workflows/build.yml)) runs bi-weekly on the 1st and 15th:
* **Smart Change Detection:** Queries upstream `ai-dock/comfyui` base digest and PyTorch nightly index. Skips rebuilds in seconds when dependencies are unchanged.
* **Parallel 4-Way Matrix Builds:** Compiles `base`, `nodes`, `blender-worker`, and `full` targets simultaneously using GitHub Actions matrix strategy.
* **Automated sm_120 & Core Validation:**
  - Verifies CUDA 12.8 runtime detection (`assert '12.8' in torch.version.cuda`).
  - Verifies Blackwell `sm_120` compute capability presence in kernel flags (`assert 'sm_120' in torch.__config__.show()`).
  - Confirms complete removal of legacy `xformers` to prevent ABI conflicts.
  - Verifies registered node mappings (`VHS_VideoCombine`, `DownloadAndLoadLivePortraitModels`, `LatentSyncNode`, `AnimateDiff`) for `nodes` and `full`.
  - Validates Blender 4.4.3 headless execution and `ComfyUI-BlenderAI-node` addon installation for `blender-worker` and `full`.
* **Enterprise Security Scanning:**
  - Audits container filesystem for leaked credentials, private keys, and `.env` files.
  - Full vulnerability and misconfiguration scan with **Aqua Security Trivy**.
  - Publishes SARIF reports to GitHub Security Tab via `github/codeql-action/upload-sarif@v4`.
  - Generates downloadable SARIF artifacts via `actions/upload-artifact@v7`.

---

## 8. Verifying Blackwell GPU Acceleration & Blender

### 1. Verify PyTorch Blackwell sm_120 Execution:
```bash
docker exec comfyui-nodes /opt/environments/python/comfyui/bin/python -c \
  "import torch; print('Device:', torch.cuda.get_device_name(0)); print('Archs:', torch.cuda.get_arch_list()); x = torch.randn(10, 10, device='cuda'); print('Tensor multiply OK:', (x @ x).shape)"
```
**Expected output:**
```text
Device: NVIDIA GeForce RTX 5090
Archs: ['sm_75', 'sm_80', 'sm_86', 'sm_90', 'sm_100', 'sm_120']
Tensor multiply OK: torch.Size([10, 10])
```

### 2. Verify ComfyUI Custom Node Registration:
```bash
docker exec comfyui-nodes /opt/environments/python/comfyui/bin/python -c \
  "import execution, nodes; nodes.init_extra_nodes(); print('Total registered nodes:', len(nodes.NODE_CLASS_MAPPINGS))"
```

### 3. Verify Blender Headless Worker:
```bash
docker exec blender-worker blender -b --version
```
**Expected output:**
```text
Blender 4.4.3 (hash 802179c51ccc built ...)
```

### 4. Verify Full Studio Image (Blender inside ComfyUI container):
```bash
docker exec comfyui-full blender -b --version
docker exec comfyui-full /opt/environments/python/comfyui/bin/python -c \
  "import execution, nodes; nodes.init_extra_nodes(); print('Total nodes:', len(nodes.NODE_CLASS_MAPPINGS))"
```

### 5. Verify ComfyUI-GGUF Quantization Engine:
```bash
docker exec comfyui-nodes /opt/environments/python/comfyui/bin/python -c \
  "import execution, nodes; nodes.init_extra_nodes(); assert any('GGUF' in k for k in nodes.NODE_CLASS_MAPPINGS); print('✔ ComfyUI-GGUF nodes verified!')"
```

---

## 9. Quantized Inference & Production Hardening

### 1. High-Performance Quantized Models (GGUF & Native FP8)
The container suite includes built-in acceleration for next-gen 12B+ diffusion models (such as **FLUX.1**, **SD 3.5**, **Wan 2.1**, and **HunyuanVideo**):
* **Native FP8 (Official ComfyUI Standard):** Download official FP8 diffusion models (`flux1-dev-fp8.safetensors`) directly into `./data/models/diffusion_models/` (or `unet/`). Zero extra nodes needed; runs natively with ~12–16GB VRAM.
* **GGUF (`ComfyUI-GGUF`):** Pre-installed in `nodes` and `full` stages. Allows loading 4-bit, 5-bit, and 8-bit quantized UNet/DiT models (e.g. `flux1-dev-Q4_K_S.gguf`) from `./data/models/unet/` using the **`Unet Loader (GGUF)`** node, enabling fast inference on 16GB cards with low memory footprints.

### 2. PyTorch Blackwell Version Guardrail (`PIP_CONSTRAINT`)
To protect your workstation against accidental PyTorch downgrades when installing community custom nodes via ComfyUI Manager:
* The container establishes `/opt/environments/python/comfyui/constraints.txt` locking the exact nightly CUDA 12.8 `sm_120` PyTorch builds.
* Enforced globally via `ENV PIP_CONSTRAINT`, preventing rogue node `requirements.txt` from replacing your Blackwell GPU runtime.

### 3. VRAM Fragmentation & Shared Memory Defense
* Configured with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` across all ComfyUI services, eliminating memory fragmentation Out-Of-Memory (OOM) errors during rapid model switching.
* Includes `--reserve-vram ${RESERVE_VRAM:-1.5}` to leave a safe headroom buffer for system display servers and external tasks.
* Allocates `shm_size: ${SHM_SIZE:-16gb}` across containers, replacing Docker's default 64MB shared memory to prevent PyTorch `Bus error` crashes during high-resolution multi-frame video decoding.

### 4. RTX 5090 (32GB) Workstation Optimization Guide

The GeForce RTX 5090 provides 32GB of high-speed GDDR7 memory (~1,792 GB/s on a 512-bit bus) paired with higher consumer boost clocks than enterprise cards (e.g. RTX PRO 6000). For models that fit within 32GB VRAM, the RTX 5090 is actually faster during sampling than a PRO 6000. For extreme multi-model pipelines, optimal results are achieved by choosing the right precision:

| Pipeline | Model Checkpoint | Text Encoder & Conditioning | VRAM Footprint | 5090 Optimization Strategy |
| :--- | :--- | :--- | :--- | :--- |
| **FLUX.1 (dev/schnell)** | `flux1-dev-fp8.safetensors` or `flux1-dev-Q8_0.gguf` | `t5xxl_fp8_e4m3fn` + `clip_l` | ~16–18 GB | **Zero offload needed.** Pipeline fits entirely in VRAM. Fastest sampling iterations. Set `COMFYUI_FLAGS=--highvram` or `--fast` in `.env` if avoiding swap delay. |
| **Wan 2.1 (1.3B)** | `wan2.1_t2v_1.3B_bf16.safetensors` | Built-in CLIP / text enc | ~8–10 GB | **Full precision (BF16).** Fits comfortably in VRAM; blazing fast text-to-video prototyping. |
| **Wan 2.1 (14B)** | `wan2.1_t2v_14B_fp8_e4m3fn.safetensors` or `wan2.1-i2v-14B-Q4_K_M.gguf` | `umt5_xxl_fp8_e4m3fn` | ~20–24 GB | **Dynamic Offload + Tiled VAE.** Default `--normalvram` automatically swaps text encoder out before DiT sampling begins. Enable **Tiled VAE Decode** (tile size 512) to prevent OOM during 81-frame latents decoding. |
| **HunyuanVideo** | `hunyuan_video_720_cfgdistill_fp8_e4m3fn.safetensors` or `Q4_K_M.gguf` | `llava` / `mllama` + `clip_l` (FP8) | ~18–22 GB | **Native PyTorch SDPA.** Blackwell `sm_120` runs scaled dot product attention without memory spikes; keep `--normalvram` active. |

> [!TIP]
> **FP8 & GGUF vs FP16 Visual Quality:** On Blackwell 5th-generation Tensor Cores, FP8 (`e4m3fn`) and GGUF (`Q8_0`/`Q4_K_M`) quantization preserve practically indistinguishable image fidelity and text alignment compared to FP16, while cutting memory footprint by 40–60%. Keep FP16 reserved for cards with $\ge$ 48–96GB VRAM.

---

## 10. Acknowledgements & Upstream Projects

* **[ComfyUI](https://github.com/comfyanonymous/ComfyUI)** by comfyanonymous — Node-based diffusion architecture.
* **[ai-dock/comfyui](https://github.com/ai-dock/comfyui)** by Rob Ballantyne — Multi-environment container runtime and process supervision.
* **[PyTorch](https://github.com/pytorch/pytorch)** — Official nightly CUDA 12.8 builds with native Blackwell `sm_120` compute kernels.
* **[Blender](https://www.blender.org/)** — Headless 3D rigging, procedural animation, and AOV pass rendering.
* **[ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF)** by City96 — High-performance quantized diffusion inference.
* **[ComfyUI-VideoHelperSuite](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite)** by Kosinkadink.
* **[ComfyUI-LivePortraitKJ](https://github.com/kijai/ComfyUI-LivePortraitKJ)** by Kijai.
* **[ComfyUI-LatentSyncWrapper](https://github.com/ShmuelRonen/ComfyUI-LatentSyncWrapper)** by Shmuel Ronen.
* **[ComfyUI-AnimateDiff-Evolved](https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved)** by Kosinkadink.
* **[ComfyUI-3D-Pack](https://github.com/MrForExample/ComfyUI-3D-Pack)** by MrForExample.
* **[ComfyUI-BlenderAI-node](https://github.com/AIGODLIKE/ComfyUI-BlenderAI-node)** by AIGODLIKE.

---

## 11. License

This project is licensed under the [MIT License](LICENSE). ComfyUI, PyTorch, Blender, and upstream packages are licensed under their respective open-source licenses.
