# ==============================================================================
# ComfyUI & Blender Docker Container Suite for NVIDIA Blackwell (RTX 5090 / 5080)
# Multi-Stage Matrix Architecture:
#   - base: ComfyUI Core + PyTorch Nightly CUDA 12.8 (sm_120) (~12 GB)
#   - nodes: Generative Video & Character Animation Stack (~16 GB)
#   - blender-worker: Lean Headless Blender 4.4 LTS OptiX/CUDA (~3.5 GB)
#   - full: All-In-One Monolith (ComfyUI + All Nodes + Blender 4.4 LTS) (~19 GB)
# ==============================================================================

# ==============================================================================
# Stage 1: Base - ComfyUI Core on PyTorch Nightly CUDA 12.8 (sm_120)
# Minimalist, clean ComfyUI core runtime without custom nodes (~12 GB)
# ==============================================================================
FROM ghcr.io/ai-dock/comfyui:latest-cuda AS base

LABEL maintainer="Tal Shmuli" \
      org.opencontainers.image.title="ComfyUI Blackwell Base (RTX 5090 / 5080)" \
      org.opencontainers.image.description="ComfyUI Core Docker Container for NVIDIA Blackwell RTX 5090 / 5080 (sm_120, CUDA 12.8)" \
      org.opencontainers.image.authors="Tal Shmuli <70902336+talshmuli@users.noreply.github.com>" \
      org.opencontainers.image.url="https://github.com/talshmuli/comfyui-blackwell-docker" \
      org.opencontainers.image.source="https://github.com/talshmuli/comfyui-blackwell-docker" \
      org.opencontainers.image.documentation="https://github.com/talshmuli/comfyui-blackwell-docker/blob/master/README.md" \
      org.opencontainers.image.vendor="Tal Shmuli" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.base.name="ghcr.io/ai-dock/comfyui:latest-cuda"

# hadolint ignore=DL3002,DL3066
USER root

# Fix Hadolint DL4006: Set the SHELL option -o pipefail before RUN with pipes
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Apply official Ubuntu security updates and eliminate dormant daemons (Caddy, Syncthing, Cloudflared)
# hadolint ignore=DL3008,DL3015
RUN apt-get update && apt-get upgrade -y && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/* && \
    rm -rf /opt/caddy /opt/syncthing /usr/bin/cloudflared \
           /opt/environments/python/jupyter \
           /opt/environments/python/serviceportal \
           /opt/environments/python/api

# Install official PyTorch, torchvision, and torchaudio nightly builds compiled
# with native CUDA 12.8 and Blackwell sm_120 compute capability kernels.
RUN /opt/environments/python/comfyui/bin/pip install --no-cache-dir \
        --force-reinstall --no-deps --pre \
        torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/nightly/cu128 && \
    /opt/environments/python/comfyui/bin/pip install --no-cache-dir \
        nvidia-cufile-cu12 \
        nvidia-cusparselt-cu12 \
        nvidia-nvshmem-cu12 && \
    /opt/environments/python/comfyui/bin/pip install --no-cache-dir --upgrade \
        "nvidia-cuda-cupti-cu12>=12.8" \
        "nvidia-cudnn-cu12>=9.20" \
        "nvidia-cublas-cu12>=12.8" \
        "nvidia-cusolver-cu12>=11.7" \
        "nvidia-cuda-runtime-cu12>=12.8" \
        "nvidia-cuda-nvrtc-cu12>=12.8" \
        "nvidia-nccl-cu12>=2.29" && \
    /opt/environments/python/comfyui/bin/pip install --no-cache-dir --upgrade \
        "pillow>=10.4.0" \
        "gitpython>=3.1.44" \
        "pyjwt>=2.9.0" \
        "certifi>=2024.8.30" \
        "h11>=0.14.0" \
        "anyio>=4.6.0" && \
    /opt/environments/python/comfyui/bin/pip uninstall -y xformers

WORKDIR /workspace

# ==============================================================================
# Stage 2: Nodes - Generative Video & 3D Character Animation Stack (~16 GB)
# Adds verified video/portrait custom nodes and pinned ML dependencies
# ==============================================================================
FROM base AS nodes

LABEL maintainer="Tal Shmuli" \
      org.opencontainers.image.title="ComfyUI Blackwell Nodes (RTX 5090 / 5080)" \
      org.opencontainers.image.description="ComfyUI Production Nodes for NVIDIA Blackwell RTX 5090 / 5080 (sm_120, CUDA 12.8)"

# Fix Hadolint DL3003: Use WORKDIR to switch directories instead of cd
WORKDIR /opt/ComfyUI/custom_nodes

# Pre-install baseline custom nodes:
# 1. VideoHelperSuite: Essential video ingestion, extraction, and mastering pipeline
# 2. LivePortraitKJ: Real-time expressive facial animation, micro-expressions, eye contact
# 3. LatentSyncWrapper: High-fidelity audio-conditioned latent phoneme lip-sync
# 4. AnimateDiff-Evolved: Industry standard for temporally coherent neural animation
# 5. ComfyUI-3D-Pack: 3D mesh rendering, UV processing, and neural reconstruction
RUN git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git && \
    git -C ComfyUI-VideoHelperSuite checkout 593fcb07 && \
    git clone --depth=1 https://github.com/kijai/ComfyUI-LivePortraitKJ.git && \
    git clone --depth=1 https://github.com/ShmuelRonen/ComfyUI-LatentSyncWrapper.git && \
    git clone https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved.git && \
    git -C ComfyUI-AnimateDiff-Evolved checkout b3e508a && \
    git clone --depth=1 https://github.com/MrForExample/ComfyUI-3D-Pack.git

# Install Python dependencies for video generation and portrait animation,
# strictly protecting PyTorch nightly CUDA 12.8 Blackwell kernels from overwrite
RUN /opt/environments/python/comfyui/bin/pip install --no-cache-dir \
        "numpy<2.0.0,>=1.24.0" \
        "protobuf<5,>=4.25.3" \
        "onnx<1.16" \
        insightface \
        "huggingface-hub<1.0" \
        "transformers<4.50,>=4.47" \
        pykalman \
        "mediapipe<0.10.15" \
        moviepy \
        ffmpeg-python \
        av \
        omegaconf \
        einops \
        onnxruntime-gpu \
        pyhocon \
        plyfile \
        trimesh && \
    /opt/environments/python/comfyui/bin/pip install --no-cache-dir --no-deps \
        "diffusers>=0.30.0,<0.31.0" \
        accelerate \
        lpips \
        torchmetrics \
        onnx2torch \
        facenet-pytorch \
        kiui

WORKDIR /workspace

# ==============================================================================
# Stage 3: Blender-Worker - Dedicated Lean Headless Blender 4.4 LTS (~3.5 GB)
# OptiX/CUDA sm_120 rendering for procedural animation & pass generation
# ==============================================================================
FROM nvidia/cuda:12.8.0-base-ubuntu24.04 AS blender-worker

LABEL maintainer="Tal Shmuli" \
      org.opencontainers.image.title="Blender Worker (RTX 5090 / 5080)" \
      org.opencontainers.image.description="Blender 4.4 LTS Headless Worker for RTX 5090 Blackwell (CUDA 12.8, OptiX)" \
      org.opencontainers.image.authors="Tal Shmuli <70902336+talshmuli@users.noreply.github.com>" \
      org.opencontainers.image.url="https://github.com/talshmuli/comfyui-blackwell-docker" \
      org.opencontainers.image.source="https://github.com/talshmuli/comfyui-blackwell-docker" \
      org.opencontainers.image.documentation="https://github.com/talshmuli/comfyui-blackwell-docker/blob/master/README.md" \
      org.opencontainers.image.vendor="Tal Shmuli" \
      org.opencontainers.image.licenses="MIT"

# Fix Hadolint DL4006: Set the SHELL option -o pipefail before RUN with pipes
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# hadolint ignore=DL3008,DL3015
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        xz-utils \
        git \
        python3 \
        libx11-6 \
        libxxf86vm1 \
        libxcursor1 \
        libxi6 \
        libxrandr2 \
        libxinerama1 \
        libgl1 \
        libegl1 \
        libsm6 \
        libxext6 \
        libxkbcommon0 \
        libdbus-1-3 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /opt/blender
RUN curl -sL "https://download.blender.org/release/Blender4.4/blender-4.4.3-linux-x64.tar.xz" | \
    tar -xJ -C /opt/blender --strip-components=1 && \
    ln -s /opt/blender/blender /usr/local/bin/blender && \
    mkdir -p /opt/blender/4.4/scripts/addons && \
    git clone --depth=1 https://github.com/AIGODLIKE/ComfyUI-BlenderAI-node.git /opt/blender/4.4/scripts/addons/ComfyUI-BlenderAI-node

WORKDIR /workspace
RUN mkdir -p /workspace/scenes /workspace/passes /opt/scripts

COPY scripts/blender_runner.py /opt/scripts/blender_runner.py
COPY scripts/blender_entrypoint.sh /opt/scripts/blender_entrypoint.sh
RUN chmod +x /opt/scripts/blender_entrypoint.sh /opt/scripts/blender_runner.py

ENTRYPOINT ["/opt/scripts/blender_entrypoint.sh"]
CMD ["python3", "/opt/scripts/blender_runner.py"]

# ==============================================================================
# Stage 4: Full (Studio Monolith) - ComfyUI + All Nodes + Blender 4.4 LTS (~19 GB)
# All-In-One Studio VFX container with ComfyUI core, custom nodes, and Blender
# ==============================================================================
FROM nodes AS full

LABEL maintainer="Tal Shmuli" \
      org.opencontainers.image.title="ComfyUI Blackwell Full Studio (RTX 5090 / 5080)" \
      org.opencontainers.image.description="All-In-One ComfyUI & Blender 4.4 LTS VFX Studio for NVIDIA Blackwell RTX 5090 / 5080 (sm_120, CUDA 12.8)" \
      org.opencontainers.image.authors="Tal Shmuli <70902336+talshmuli@users.noreply.github.com>" \
      org.opencontainers.image.url="https://github.com/talshmuli/comfyui-blackwell-docker" \
      org.opencontainers.image.source="https://github.com/talshmuli/comfyui-blackwell-docker" \
      org.opencontainers.image.documentation="https://github.com/talshmuli/comfyui-blackwell-docker/blob/master/README.md" \
      org.opencontainers.image.vendor="Tal Shmuli" \
      org.opencontainers.image.licenses="MIT"

# Fix Hadolint DL4006: Set the SHELL option -o pipefail before RUN with pipes
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# hadolint ignore=DL3008,DL3015
RUN apt-get update && apt-get install -y --no-install-recommends \
        libx11-6 \
        libxxf86vm1 \
        libxcursor1 \
        libxi6 \
        libxrandr2 \
        libxinerama1 \
        libgl1 \
        libegl1 \
        libsm6 \
        libxext6 \
        libxkbcommon0 \
        libdbus-1-3 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /opt/blender
RUN curl -sL "https://download.blender.org/release/Blender4.4/blender-4.4.3-linux-x64.tar.xz" | \
    tar -xJ -C /opt/blender --strip-components=1 && \
    ln -s /opt/blender/blender /usr/local/bin/blender && \
    mkdir -p /opt/blender/4.4/scripts/addons && \
    git clone --depth=1 https://github.com/AIGODLIKE/ComfyUI-BlenderAI-node.git /opt/blender/4.4/scripts/addons/ComfyUI-BlenderAI-node

WORKDIR /workspace

