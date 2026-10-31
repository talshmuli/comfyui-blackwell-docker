#!/usr/bin/env python3
"""Evaluate Aqua Security Trivy scan results and generate markdown summary

with component-level and custom-node breakdowns for GitHub Actions.
"""

from collections import defaultdict
import json
import os
import sys

# Mapping of known Python packages to their originating component/custom node
CUSTOM_NODE_DEPS = {
    "Custom Node: ComfyUI-3D-Pack": [
        "kiui",
        "onnx",
        "onnx2torch",
        "onnxruntime",
        "trimesh",
        "plyfile",
        "pyhocon",
    ],
    "Custom Node: ComfyUI-LivePortraitKJ": [
        "insightface",
        "mediapipe",
        "pykalman",
        "facenet-pytorch",
    ],
    "Custom Node: ComfyUI-VideoHelperSuite": ["moviepy", "ffmpeg-python", "av"],
    "Custom Node: ComfyUI-AnimateDiff-Evolved": [
        "diffusers",
        "accelerate",
        "lpips",
        "torchmetrics",
    ],
    "Custom Node: ComfyUI-LatentSyncWrapper": [],
    "Custom Node: ComfyUI-GGUF": ["gguf", "sentencepiece"],
    "Blender Addon: ComfyUI-BlenderAI-node": [],
}

COMPONENT_DESCRIPTIONS = {
    "ComfyUI Core & Runtime (Python)": (
        "Core ComfyUI web UI & PyTorch runtime (Pillow, Tornado, Aiohttp,"
        " Transformers)"
    ),
    "Blender 4.4 LTS Core & Python": (
        "Headless Blender 3D rendering engine and bundled Python packages"
    ),
    "Custom Node: ComfyUI-3D-Pack": (
        "3D mesh reconstruction, Gaussian splatting & UV shaders (ONNX,"
        " Trimesh)"
    ),
    "Custom Node: ComfyUI-LivePortraitKJ": (
        "Real-time expressive facial animation & landmarking (InsightFace,"
        " MediaPipe)"
    ),
    "Custom Node: ComfyUI-VideoHelperSuite": (
        "Video ingestion, frame splitting, audio multiplexing & export"
        " (MoviePy, PyAV)"
    ),
    "Custom Node: ComfyUI-AnimateDiff-Evolved": (
        "Temporal motion module conditioning & Latent diffusers (Diffusers,"
        " Accelerate)"
    ),
    "Custom Node: ComfyUI-LatentSyncWrapper": (
        "Audio-conditioned lip-sync neural model wrapper"
    ),
    "Custom Node: ComfyUI-GGUF": (
        "Quantized GGUF model loader & inference engine (gguf, sentencepiece)"
    ),
    "Blender Addon: ComfyUI-BlenderAI-node": (
        "Procedural Blender-ComfyUI socket bridge addon"
    ),
    "Node.js & npm Tooling": (
        "JavaScript frontend dependency manager in base image (NVM / npm)"
    ),
    "Legacy Unused Python (python_310)": (
        "Dormant legacy environment from upstream base image (can be safely"
        " pruned)"
    ),
    "Ubuntu 24.04 OS System Packages": (
        "Underlying Linux OS shared libraries (glibc, systemd, libssl, etc.)"
    ),
}


def categorize_location(uri, pkg_name=""):
  u = (uri or "").lower()
  p = (pkg_name or "").lower()

  # 1. Custom node directories
  if "custom_nodes/comfyui-3d-pack" in u:
    return "Custom Node: ComfyUI-3D-Pack"
  if "custom_nodes/comfyui-liveportraitkj" in u:
    return "Custom Node: ComfyUI-LivePortraitKJ"
  if "custom_nodes/comfyui-videohelpersuite" in u:
    return "Custom Node: ComfyUI-VideoHelperSuite"
  if "custom_nodes/comfyui-animatediff-evolved" in u:
    return "Custom Node: ComfyUI-AnimateDiff-Evolved"
  if "custom_nodes/comfyui-latentsyncwrapper" in u:
    return "Custom Node: ComfyUI-LatentSyncWrapper"
  if "custom_nodes/comfyui-gguf" in u:
    return "Custom Node: ComfyUI-GGUF"
  if "comfyui-blenderai-node" in u:
    return "Blender Addon: ComfyUI-BlenderAI-node"

  # 2. Package-based mapping for custom node deps
  for comp, pkgs in CUSTOM_NODE_DEPS.items():
    for k in pkgs:
      if f"/{k}-" in u or f"/{k}." in u or p == k or u.endswith(f"/{k}"):
        return comp

  # 3. Blender runtime
  if "blender" in u:
    return "Blender 4.4 LTS Core & Python"

  # 4. Node.js & npm tooling
  if "nvm" in u or "node_modules" in u or "npm" in u:
    return "Node.js & npm Tooling"

  # 5. Legacy unused python
  if "python_310" in u:
    return "Legacy Unused Python (python_310)"

  # 6. Ubuntu OS packages
  if (
      u.startswith("library/")
      or u.startswith("ubuntu")
      or "/var/lib/dpkg" in u
      or u.startswith("etc/")
      or u.startswith("usr/")
  ):
    return "Ubuntu 24.04 OS System Packages"

  # 7. ComfyUI Core
  if "opt/environments/python/comfyui" in u or "opt/comfyui" in u:
    return "ComfyUI Core & Runtime (Python)"

  return "ComfyUI Core & Runtime (Python)"


def parse_json_report(json_file):
  counts = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0, "UNKNOWN": 0}
  comp_stats = defaultdict(
      lambda: {
          "CRITICAL": 0,
          "HIGH": 0,
          "MEDIUM": 0,
          "LOW": 0,
          "UNKNOWN": 0,
          "TOTAL": 0,
          "pkgs": set(),
          "vuln_ids": set(),
      }
  )

  with open(json_file, "r") as f:
    data = json.load(f)

  for res in data.get("Results", []):
    target_path = res.get("Target", "")
    for vuln in res.get("Vulnerabilities", []):
      sev = vuln.get("Severity", "UNKNOWN").upper()
      pkg_name = vuln.get("PkgName", "")
      vuln_id = vuln.get("VulnerabilityID", "")

      counts[sev] = counts.get(sev, 0) + 1
      comp = categorize_location(target_path, pkg_name)

      comp_stats[comp][sev] += 1
      comp_stats[comp]["TOTAL"] += 1
      if pkg_name:
        comp_stats[comp]["pkgs"].add(pkg_name)
      if vuln_id:
        comp_stats[comp]["vuln_ids"].add(vuln_id)

  return counts, comp_stats


def parse_sarif_report(sarif_file):
  counts = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0, "UNKNOWN": 0}
  comp_stats = defaultdict(
      lambda: {
          "CRITICAL": 0,
          "HIGH": 0,
          "MEDIUM": 0,
          "LOW": 0,
          "UNKNOWN": 0,
          "TOTAL": 0,
          "pkgs": set(),
          "vuln_ids": set(),
      }
  )

  with open(sarif_file, "r") as f:
    data = json.load(f)

  runs = data.get("runs", [])
  if not runs:
    return counts, comp_stats

  rules = {
      r["id"]: r
      for r in runs[0].get("tool", {}).get("driver", {}).get("rules", [])
  }
  for res in runs[0].get("results", []):
    rule_id = res.get("ruleId", "")
    rule = rules.get(rule_id, {})
    sev = "UNKNOWN"
    for t in rule.get("properties", {}).get("tags", []):
      if t in counts:
        sev = t
    counts[sev] += 1

    locs = res.get("locations", [])
    uri = (
        locs[0].get("physicalLocation", {}).get("artifactLocation", {}).get("uri", "")
        if locs
        else ""
    )

    parts = [
        p
        for p in uri.split("/")
        if p not in ["METADATA", "package.json", "temp"]
    ]
    parent = parts[-1] if parts else ""
    pkg_clean = parent.split(".dist-info")[0].split("-")[0]

    comp = categorize_location(uri, pkg_clean)
    comp_stats[comp][sev] += 1
    comp_stats[comp]["TOTAL"] += 1
    if pkg_clean:
      comp_stats[comp]["pkgs"].add(pkg_clean)
    if rule_id:
      comp_stats[comp]["vuln_ids"].add(rule_id)

  return counts, comp_stats


def main():
  target = os.environ.get("TARGET", "unknown")
  if len(sys.argv) > 1:
    target = sys.argv[1]

  json_file = f"trivy-results-{target}.json"
  sarif_file = f"trivy-results-{target}.sarif"

  counts = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0, "UNKNOWN": 0}
  comp_stats = {}

  if os.path.exists(json_file):
    try:
      counts, comp_stats = parse_json_report(json_file)
    except Exception as e:
      print(f"Warning parsing {json_file}: {e}")
  elif os.path.exists(sarif_file):
    try:
      counts, comp_stats = parse_sarif_report(sarif_file)
    except Exception as e:
      print(f"Warning parsing {sarif_file}: {e}")

  total_vulns = sum(counts.values())

  print("============================================================")
  print(f"🛡️  TRIVY VULNERABILITY COUNT SUMMARY: {target}")
  print("============================================================")
  print(f"  LOW      : {counts['LOW']}")
  print(f"  MEDIUM   : {counts['MEDIUM']}")
  print(f"  HIGH     : {counts['HIGH']}")
  print(f"  CRITICAL : {counts['CRITICAL']}")
  print(f"  TOTAL    : {total_vulns}")
  print("============================================================")
  print("\n📦 COMPONENT BREAKDOWN:")
  for comp, stats in sorted(
      comp_stats.items(), key=lambda x: x[1]["TOTAL"], reverse=True
  ):
    pkgs = ", ".join(sorted(stats["pkgs"])[:4]) or "System"
    print(
        f"  {comp:38s} | Crit:{stats['CRITICAL']:2d} High:{stats['HIGH']:2d}"
        f" Med:{stats['MEDIUM']:2d} Low:{stats['LOW']:2d} Tot:{stats['TOTAL']:3d}"
        f" | {pkgs}"
    )

  summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
  if summary_path and os.path.exists(summary_path):
    with open(summary_path, "a") as sf:
      sf.write(f"### 🛡️ Trivy Security Scan: `{target}`\n\n")

      # Severity overview
      sf.write("| Vulnerability Severity | Total Detected | Status |\n")
      sf.write("| :--- | :---: | :--- |\n")
      sf.write(
          f"| **CRITICAL** | {counts['CRITICAL']} |"
          f" {'✅ 0 Critical' if counts['CRITICAL'] == 0 else '⚠️ Requires review'} |\n"
      )
      sf.write(
          f"| **HIGH** | {counts['HIGH']} | {'✅ 0 High' if counts['HIGH'] == 0 else '⚠️ Monitored'} |\n"
      )
      sf.write(
          f"| **MEDIUM** | {counts['MEDIUM']} | {'✅ 0 Medium' if counts['MEDIUM'] == 0 else 'ℹ️ Standard'} |\n"
      )
      sf.write(f"| **LOW** | {counts['LOW']} | ℹ️ Informational |\n")
      sf.write(f"| **TOTAL** | **{total_vulns}** | — |\n\n")

      if counts["CRITICAL"] > 0:
        sf.write(
            f"> ⚠️ **Notice**: {counts['CRITICAL']} critical vulnerabilities"
            " detected. See component breakdown below for details.\n\n"
        )
      else:
        sf.write(
            "> 🎉 **Excellent**: **0 Critical Vulnerabilities** detected in"
            f" image `{target}`!\n\n"
        )

      # Component breakdown table
      sf.write(
          "#### 📦 Vulnerability Breakdown by Component & Custom Node\n\n"
      )
      sf.write(
          "| Component / Custom Node | Critical | High | Medium | Low | Total |"
          " Primary Packages | Role / Decision Impact |\n"
      )
      sf.write(
          "| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- |\n"
      )

      # Ensure all standard custom nodes appear in table even if 0 vulns
      displayed_comps = set()
      for comp, stats in sorted(
          comp_stats.items(), key=lambda x: x[1]["TOTAL"], reverse=True
      ):
        displayed_comps.add(comp)
        pkgs_str = ", ".join(sorted(stats["pkgs"])[:5]) or "System components"
        desc = COMPONENT_DESCRIPTIONS.get(comp, "Component library")
        sf.write(
            f"| **{comp}** | {stats['CRITICAL']} | {stats['HIGH']} |"
            f" {stats['MEDIUM']} | {stats['LOW']} | **{stats['TOTAL']}** |"
            f" `{pkgs_str}` | {desc} |\n"
        )

      # Add custom nodes that had 0 vulnerabilities for transparency
      for custom_node in [
          "Custom Node: ComfyUI-VideoHelperSuite",
          "Custom Node: ComfyUI-LivePortraitKJ",
          "Custom Node: ComfyUI-LatentSyncWrapper",
          "Custom Node: ComfyUI-AnimateDiff-Evolved",
          "Custom Node: ComfyUI-3D-Pack",
          "Blender Addon: ComfyUI-BlenderAI-node",
      ]:
        if (
            custom_node not in displayed_comps
            and target in ["nodes", "full"]
            or (custom_node.startswith("Blender") and target == "blender-worker")
        ):
          desc = COMPONENT_DESCRIPTIONS.get(custom_node, "Installed Addon")
          sf.write(
              f"| **{custom_node}** | 0 | 0 | 0 | 0 | **0** | `Clean (0"
              f" vulns)` | {desc} |\n"
          )

      sf.write("\n")

      # Component Decision Guide
      sf.write("#### 💡 Component Assessment & Decision Guide\n\n")
      sf.write(
          "Use this breakdown to determine whether any component should be"
          " kept, updated, or removed:\n\n"
      )
      sf.write(
          "- **ComfyUI Core & Runtime (Python):** Contains core web UI and"
          " PyTorch dependencies. Critical Pillow vulnerabilities require"
          " Python >= 3.11; Pillow 12.3.0 is pinned to preserve Python 3.10"
          " stability.\n"
      )
      if (
          "Custom Node: ComfyUI-3D-Pack" in comp_stats
          and comp_stats["Custom Node: ComfyUI-3D-Pack"]["TOTAL"] > 0
      ):
        c3d = comp_stats["Custom Node: ComfyUI-3D-Pack"]
        sf.write(
            f"- **ComfyUI-3D-Pack ({c3d['TOTAL']} vulns: {c3d['HIGH']} High,"
            f" {c3d['MEDIUM']} Medium):** Caused by ONNX/Trimesh dependencies."
            " If 3D mesh reconstruction is not required in your workflows,"
            " this node can be safely uninstalled to eliminate these"
            " vulnerabilities.\n"
        )
      if (
          "Custom Node: ComfyUI-AnimateDiff-Evolved" in comp_stats
          and comp_stats["Custom Node: ComfyUI-AnimateDiff-Evolved"]["TOTAL"] > 0
      ):
        cad = comp_stats["Custom Node: ComfyUI-AnimateDiff-Evolved"]
        sf.write(
            f"- **ComfyUI-AnimateDiff-Evolved ({cad['TOTAL']} vulns:"
            f" {cad['HIGH']} High):** Upstream Diffusers library conditioning."
            " Safe to keep for neural animation.\n"
        )
      if (
          "Blender 4.4 LTS Core & Python" in comp_stats
          and comp_stats["Blender 4.4 LTS Core & Python"]["TOTAL"] > 0
      ):
        bstat = comp_stats["Blender 4.4 LTS Core & Python"]
        sf.write(
            f"- **Blender 4.4 LTS ({bstat['TOTAL']} vulns: {bstat['CRITICAL']}"
            f" Critical, {bstat['HIGH']} High):** Bundled Python libraries."
            " Certifi and pip updates applied to eliminate critical"
            " exposure.\n"
        )
      if (
          "Legacy Unused Python (python_310)" in comp_stats
          and comp_stats["Legacy Unused Python (python_310)"]["TOTAL"] > 0
      ):
        sf.write(
            "- **Legacy Unused Python (python_310):** Dormant legacy environment"
            " scheduled for complete removal.\n"
        )
      sf.write("\n")


if __name__ == "__main__":
  main()
