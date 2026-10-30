#!/usr/bin/env python3
"""Evaluate Aqua Security Trivy scan results and generate markdown summary."""

import json
import os
import sys


def main():
    target = os.environ.get("TARGET", "unknown")
    if len(sys.argv) > 1:
        target = sys.argv[1]

    json_file = f"trivy-results-{target}.json"
    counts = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0, "UNKNOWN": 0}

    if os.path.exists(json_file):
        try:
            with open(json_file, "r") as f:
                data = json.load(f)
            for res in data.get("Results", []):
                for vuln in res.get("Vulnerabilities", []):
                    sev = vuln.get("Severity", "UNKNOWN").upper()
                    counts[sev] = counts.get(sev, 0) + 1
        except Exception as e:
            print(f"Warning parsing {json_file}: {e}")

    print("============================================================")
    print(f"🛡️  TRIVY VULNERABILITY COUNT SUMMARY: {target}")
    print("============================================================")
    print(f"  LOW      : {counts['LOW']}")
    print(f"  MEDIUM   : {counts['MEDIUM']}")
    print(f"  HIGH     : {counts['HIGH']}")
    print(f"  CRITICAL : {counts['CRITICAL']}")
    print("============================================================")

    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path and os.path.exists(summary_path):
        with open(summary_path, "a") as sf:
            sf.write(f"### 🛡️ Trivy Security Scan: `{target}`\n\n")
            sf.write("| Vulnerability Severity | Total Detected |\n")
            sf.write("| :--- | :---: |\n")
            sf.write(f"| **LOW** | {counts['LOW']} |\n")
            sf.write(f"| **MEDIUM** | {counts['MEDIUM']} |\n")
            sf.write(f"| **HIGH** | {counts['HIGH']} |\n")
            sf.write(f"| **CRITICAL** | {counts['CRITICAL']} |\n\n")
            if counts["CRITICAL"] > 0:
                sf.write(f"> ⚠️ **Notice**: {counts['CRITICAL']} critical vulnerabilities detected from upstream packages.\n\n")


if __name__ == "__main__":
    main()
