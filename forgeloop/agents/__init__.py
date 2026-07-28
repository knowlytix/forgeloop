"""agentlab — companion library for Building Agentic AI Systems from Scratch.

Sub-packages are introduced chapter by chapter. See outline.tex for the map.
"""

from forgeloop._paths import data_path, data_root, set_data_dir
from forgeloop.artifacts import ensure_artifacts, missing_artifacts

__version__ = "0.1.0"

__all__ = [
    "data_root",
    "data_path",
    "set_data_dir",
    "ensure_artifacts",
    "missing_artifacts",
    "__version__",
]
