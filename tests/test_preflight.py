from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


_path = Path(__file__).parents[1] / "scripts" / "preflight.py"
_spec = spec_from_file_location("preflight", _path)
preflight = module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(preflight)


def test_accepts_nvidia_alongside_llvmpipe():
    output = """GPU0:\n\tdeviceName         = NVIDIA RTX A4000\n\tdriverName         = NVIDIA\nGPU1:\n\tdeviceName         = llvmpipe\n\tdeviceType         = PHYSICAL_DEVICE_TYPE_CPU\n"""
    assert preflight.has_nvidia_vulkan_device(output)


def test_rejects_cpu_only_vulkan():
    output = """GPU0:\n\tdeviceName         = llvmpipe\n\tdriverName         = llvmpipe\n\tdeviceType         = PHYSICAL_DEVICE_TYPE_CPU\n"""
    assert not preflight.has_nvidia_vulkan_device(output)
