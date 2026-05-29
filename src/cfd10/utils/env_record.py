"""Record environment information for experiment reproducibility."""

from __future__ import annotations

import platform
from typing import Dict


def record_env() -> Dict[str, str]:
    """Capture interpreter / platform / torch-GPU info for run provenance.

    Returns:
        A dict of environment facts safe to serialise into run metadata.
    """
    info: Dict[str, str] = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
    }
    try:
        import torch

        info["torch_version"] = torch.__version__
        info["cuda_version"] = str(torch.version.cuda)
        info["gpu"] = (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
        )
    except ImportError:
        info["torch_version"] = "N/A"
    return info
