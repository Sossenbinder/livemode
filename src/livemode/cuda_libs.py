"""Load the CUDA 12 libraries that faster-whisper (CTranslate2) needs.

The system can have a different CUDA version. The nvidia-cublas-cu12 and
nvidia-cudnn-cu12 packages put the CUDA 12 libraries into the virtual environment,
but the dynamic loader does not search there. This module loads them globally
before CTranslate2 starts, so its lookups by name find them.
"""

import ctypes
import importlib.util
from pathlib import Path


def preload() -> None:
    spec = importlib.util.find_spec("nvidia")
    if spec is None or not spec.submodule_search_locations:
        return
    libs = [
        lib
        for root in spec.submodule_search_locations
        for lib in sorted(Path(root).glob("*/lib/lib*.so.*"))
    ]
    # Some libraries need others that are not loaded yet. Repeat until no
    # library loads in a full pass.
    while libs:
        failed = []
        for lib in libs:
            try:
                ctypes.CDLL(str(lib), mode=ctypes.RTLD_GLOBAL)
            except OSError:
                failed.append(lib)
        if len(failed) == len(libs):
            break
        libs = failed
