#!/usr/bin/env python3
"""Fail-fast host/runtime checks before spending the evaluation budget."""

from __future__ import annotations

import json
import os
import subprocess


def has_nvidia_vulkan_device(output: str) -> bool:
    """Return true when any reported Vulkan device is backed by NVIDIA.

    A host may expose both NVIDIA and Mesa llvmpipe.  The presence of a CPU
    device must not invalidate a usable NVIDIA device.
    """
    blocks = output.split("\nGPU")
    return any(
        "deviceName" in block
        and ("driverName         = NVIDIA" in block or "vendorID           = 0x10de" in block)
        for block in blocks
    )


def main() -> int:
    capabilities = set(os.environ.get("NVIDIA_DRIVER_CAPABILITIES", "").split(","))
    vulkan = subprocess.run(["vulkaninfo", "--summary"], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    output = vulkan.stdout
    gpu_vulkan = vulkan.returncode == 0 and has_nvidia_vulkan_device(output)
    report = {
        "nvidia_driver_capabilities": sorted(x for x in capabilities if x),
        "graphics_capability": "graphics" in capabilities or "all" in capabilities,
        "vulkan_nvidia_gpu": gpu_vulkan,
        "vulkan_returncode": vulkan.returncode,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    if not gpu_vulkan:
        print("BLOCKED: no usable NVIDIA Vulkan physical device")
        return 2
    if not report["graphics_capability"]:
        print("WARNING: graphics is absent from NVIDIA_DRIVER_CAPABILITIES, but the runtime Vulkan probe succeeded")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
