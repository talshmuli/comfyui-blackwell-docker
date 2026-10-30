#!/usr/bin/env python3
"""Check for Upstream ComfyUI & PyTorch Nightly Updates.

Evaluates upstream base image digest on GHCR and latest PyTorch nightly CUDA 12.8
wheels to determine if a new container image build is required.
"""

import os
import re
import subprocess
import sys
import urllib.request


def main():
    should_build = False
    reasons = []

    # 1. Check upstream base image digest
    upstream_digest = ""
    try:
        cmd_up = [
            "docker", "run", "--rm", "quay.io/skopeo/stable",
            "inspect", "docker://ghcr.io/ai-dock/comfyui:latest-cuda",
            "--format", "{{.Digest}}",
        ]
        upstream_digest = subprocess.check_output(cmd_up, timeout=60).decode().strip()
    except Exception as e:
        print(f"Warning: Failed to fetch upstream digest: {e}")

    published_base_digest = ""
    published_tags = ""
    image_name = os.environ.get("IMAGE_NAME", "")

    if image_name:
        try:
            cmd_pub = [
                "docker", "run", "--rm", "quay.io/skopeo/stable",
                "inspect", f"docker://ghcr.io/{image_name}:latest",
                "--format", '{{index .Labels "upstream.base.digest"}}',
            ]
            published_base_digest = subprocess.check_output(cmd_pub, timeout=60).decode().strip()
        except Exception:
            published_base_digest = ""

        try:
            cmd_tags = [
                "docker", "run", "--rm", "quay.io/skopeo/stable",
                "inspect", f"docker://ghcr.io/{image_name}:latest",
                "--format", "{{.RepoTags}}",
            ]
            published_tags = subprocess.check_output(cmd_tags, timeout=60).decode().strip()
        except Exception:
            published_tags = ""

    if upstream_digest and published_base_digest and upstream_digest != published_base_digest:
        should_build = True
        reasons.append(f"Upstream ComfyUI base image updated: {upstream_digest} != {published_base_digest}")
    elif upstream_digest and not published_base_digest:
        print("No base digest label in published image; checking tags and packages...")

    # 2. Check latest PyTorch nightly release
    latest_torch = ""
    try:
        req = urllib.request.Request(
            "https://download.pytorch.org/whl/nightly/cu128/torch/",
            headers={"User-Agent": "Mozilla/5.0"},
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            html = resp.read().decode("utf-8")
        matches = re.findall(r"torch-2\.[0-9.]*\.dev[0-9]*", html)
        latest_torch = matches[-1] if matches else ""
    except Exception as e:
        print(f"Warning: Failed to fetch PyTorch nightly index: {e}")

    if latest_torch and latest_torch not in published_tags:
        should_build = True
        reasons.append(f"New PyTorch nightly cu128 wheel detected: {latest_torch} not present in published tags")

    print("=== Update Detection Summary ===")
    print(f"Upstream Base Digest:  {upstream_digest}")
    print(f"Published Base Digest: {published_base_digest}")
    print(f"Latest PyTorch Nightly: {latest_torch}")
    print(f"Published Tags:        {published_tags}")
    print(f"Should Build:          {should_build}")

    reason_str = "; ".join(reasons) if reasons else "No upstream or package updates detected"
    print(f"Decision: {reason_str}")

    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a") as f:
            f.write(f"should_build={'true' if should_build else 'false'}\n")
            f.write(f"reason={reason_str}\n")
            f.write(f"upstream_digest={upstream_digest}\n")
            f.write(f"latest_torch={latest_torch}\n")


if __name__ == "__main__":
    main()
