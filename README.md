# ComfyUI & Blender VFX Studio Docker for NVIDIA Blackwell (RTX 5090 / 5080 - sm_120)

[![Build & Publish ComfyUI Blackwell Image](https://github.com/talshmuli/comfyui-blackwell-docker/actions/workflows/build.yml/badge.svg)](https://github.com/talshmuli/comfyui-blackwell-docker/actions/workflows/build.yml)
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

## 2. 4-Tier Multi-Stage Image Matrix

| Target Stage | Canonical Tag | Estimated Size | Included Components & Features | Primary Use Case |
| :--- | :--- | :--- | :--- | :--- |
| **`base`** | `base-cuda128-sm120` | ~12 GB | • Upstream `ai-dock/comfyui:latest-cuda`<br>• Official PyTorch Nightly CUDA 12.8 (`sm_120`)<br>• CUDA 12.8 driver runtime (`cudnn>=9.20`, `cublas>=12.8`)<br>• Clean ComfyUI core (0 custom nodes)<br>• Incompatible `xformers` uninstalled | Minimalist runtime, core testing, building custom node sets from scratch. |
| **`nodes`** *(Default)* | `nodes-cuda128-sm120`<br>`latest` | ~16 GB | • All `base` components<br>• `ComfyUI-VideoHelperSuite` (pinned `593fcb07`)<br>• `ComfyUI-LivePortraitKJ` (Facial dynamics)<br>• `ComfyUI-LatentSyncWrapper` (Audio lip-sync)<br>• `ComfyUI-AnimateDiff-Evolved` (pinned `b3e508a`)<br>• `ComfyUI-3D-Pack` (Mesh & UV processing)<br>• Pinned ML packages (`diffusers<0.31`, `insightface`, `mediapipe`) | Production generative video, character animation, facial performance capture. |
| **`blender-worker`** | `blender-worker`<br>`blender-4.4-worker` | ~3.5 GB | • Minimal `nvidia/cuda:12.8.0-base-ubuntu24.04`<br>• Official Blender 4.4.3 LTS Headless<br>• Native OptiX & CUDA Blackwell rendering<br>• `ComfyUI-BlenderAI-node` pre-installed<br>• Automated batch render runner (`blender_runner.py`) | Headless geometric pass generation (Depth, Normal, OpenPose), procedural 3D rigging. Pairs with `nodes` via shared volume. |
| **`full`** | `full-cuda128-sm120` | ~19 GB | • All `nodes` components<br>• Official Blender 4.4.3 LTS Headless<br>• `ComfyUI-BlenderAI-node` addon<br>• Native OptiX & CUDA Blackwell rendering | All-in-one single-container studio — ComfyUI + all custom nodes + Blender 4.4 in one image. Ideal for single-GPU workstations. |

---

## 3. Quickstart & Docker Compose Orchestration

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

### Custom Ports & Configuration
To avoid port conflicts with host services, define custom ports in your environment:
```bash
COMFYUI_PORT=8189 docker compose --profile modular up -d
```

---

## 4. Compose Configuration Specification

The complete production orchestration definition (`docker-compose.yml`):

```yaml
services:
  comfyui:
    image: ${COMFYUI_IMAGE:-ghcr.io/talshmuli/comfyui-blackwell-docker:nodes-cuda128-sm120}
    container_name: comfyui-nodes
    profiles: ["modular", "nodes"]
    restart: unless-stopped
    ports:
      - "${COMFYUI_PORT:-8188}:8188"
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - NVIDIA_DRIVER_CAPABILITIES=all
      - COMFYUI_FLAGS=--preview-method auto --normalvram
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
    networks:
      - vfx-net

  blender-worker:
    image: ${BLENDER_IMAGE:-ghcr.io/talshmuli/comfyui-blackwell-docker:blender-worker}
    container_name: blender-worker
    profiles: ["modular", "dev"]
    restart: unless-stopped
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
    image: ${COMFYUI_FULL_IMAGE:-ghcr.io/talshmuli/comfyui-blackwell-docker:full-cuda128-sm120}
    container_name: comfyui-full
    profiles: ["monolith", "full"]
    restart: unless-stopped
    ports:
      - "${COMFYUI_PORT:-8188}:8188"
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - NVIDIA_DRIVER_CAPABILITIES=all
      - COMFYUI_FLAGS=--preview-method auto --normalvram
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

## 5. Strict Semantic Versioning & Immutability

To guarantee reproducibility in production studio environments, images are published with strict version tagging:

1. **Immutable Exact SemVer Tags (Never overwritten):**
   * `base`: `ghcr.io/talshmuli/comfyui-blackwell-docker:v0.2.2-torch2.12.0.dev-cu128-base-YYYYMMDD-r<RUN_ID>`
   * `nodes`: `ghcr.io/talshmuli/comfyui-blackwell-docker:v0.2.2-torch2.12.0.dev-cu128-nodes-YYYYMMDD-r<RUN_ID>`
   * `full`: `ghcr.io/talshmuli/comfyui-blackwell-docker:v0.2.2-torch2.12.0.dev-cu128-full-YYYYMMDD-r<RUN_ID>`
   * `blender-worker`: `ghcr.io/talshmuli/comfyui-blackwell-docker:blender4.4.3-cuda128-YYYYMMDD-r<RUN_ID>`
2. **Canonical Series Tags (Updated per stable release):**
   * `base-cuda128-sm120`
   * `nodes-cuda128-sm120` (also alias `v0.2.2-cuda128-sm120`)
   * `full-cuda128-sm120`
   * `blender-worker`
3. **Latest Tag:**
   * `ghcr.io/talshmuli/comfyui-blackwell-docker:latest` points directly to the `nodes` production workstation.

---

## 6. Automated CI Pipeline & Verification

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

## 7. Verifying Blackwell GPU Acceleration & Blender

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

---

## 8. Acknowledgements & Upstream Projects

* **[ComfyUI](https://github.com/comfyanonymous/ComfyUI)** by comfyanonymous — Node-based diffusion architecture.
* **[ai-dock/comfyui](https://github.com/ai-dock/comfyui)** by Rob Ballantyne — Multi-environment container runtime and process supervision.
* **[PyTorch](https://github.com/pytorch/pytorch)** — Official nightly CUDA 12.8 builds with native Blackwell `sm_120` compute kernels.
* **[Blender](https://www.blender.org/)** — Headless 3D rigging, procedural animation, and AOV pass rendering.
* **[ComfyUI-VideoHelperSuite](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite)** by Kosinkadink.
* **[ComfyUI-LivePortraitKJ](https://github.com/kijai/ComfyUI-LivePortraitKJ)** by Kijai.
* **[ComfyUI-LatentSyncWrapper](https://github.com/ShmuelRonen/ComfyUI-LatentSyncWrapper)** by Shmuel Ronen.
* **[ComfyUI-AnimateDiff-Evolved](https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved)** by Kosinkadink.
* **[ComfyUI-3D-Pack](https://github.com/MrForExample/ComfyUI-3D-Pack)** by MrForExample.
* **[ComfyUI-BlenderAI-node](https://github.com/AIGODLIKE/ComfyUI-BlenderAI-node)** by AIGODLIKE.

---

## 9. License

This project is licensed under the [MIT License](LICENSE). ComfyUI, PyTorch, Blender, and upstream packages are licensed under their respective open-source licenses.
