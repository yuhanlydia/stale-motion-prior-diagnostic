#!/usr/bin/env python3
"""Fail-fast host/runtime checks before spending the evaluation budget."""

from __future__ import annotations

import json
import os
import subprocess


def main() -> int:
    capabilities = set(os.environ.get("NVIDIA_DRIVER_CAPABILITIES", "").split(","))
    vulkan = subprocess.run(["vulkaninfo", "--summary"], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    output = vulkan.stdout
    gpu_vulkan = "NVIDIA" in output and "PHYSICAL_DEVICE_TYPE_CPU" not in output
    report = {
        "nvidia_driver_capabilities": sorted(x for x in capabilities if x),
        "graphics_capability": "graphics" in capabilities or "all" in capabilities,
        "vulkan_nvidia_gpu": gpu_vulkan,
        "vulkan_returncode": vulkan.returncode,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["graphics_capability"] or not gpu_vulkan:
        print("BLOCKED: restart container with NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

