"""forgeloop — companion library for the *Beyond Prompt / Ship / Chunk and Pray* trilogy.

One installable package spanning the three books' companion code:

- :mod:`forgeloop.agents`   — building governed agents (*Beyond Prompt and Pray*)
- :mod:`forgeloop.testing`  — the GMS test framework (*Beyond Ship and Pray*)
- :mod:`forgeloop.rag`      — governed retrieval (*Beyond Chunk and Pray*)

Shared across them: :func:`data_root`/:func:`data_path`/:func:`set_data_dir`
resolve a book's ``data/`` directory, while :func:`missing_artifacts` and
:func:`build_instructions` report which large trained artifacts are not built
yet and what to run to build them (they are trained locally, not downloaded).
"""

from importlib.metadata import PackageNotFoundError, version as _version
from pathlib import Path

from forgeloop._paths import data_path, data_root, set_data_dir
from forgeloop.artifacts import build_instructions, ensure_artifacts, missing_artifacts

# Read from installed package metadata rather than hardcoding, so pyproject.toml
# is the single source of truth. A hardcoded literal here silently drifts: it
# said 0.2.3 while pyproject said 0.2.4, and the docs build reports
# forgeloop.__version__ as the site version.
try:
    __version__ = _version("forgeloop")
except PackageNotFoundError:  # running from a checkout without an install
    __version__ = "0.0.0+unknown"


def notebooks_dir() -> Path:
    """Path to the bundled chapter notebooks shipped with the package.

    Contains one sub-directory per book (``beyond-prompt-and-pray`` etc.). Copy
    them somewhere writable to run them, e.g.::

        import shutil, forgeloop
        shutil.copytree(forgeloop.notebooks_dir(), "./forgeloop-notebooks")
    """
    return Path(__file__).resolve().parent / "notebooks"


__all__ = [
    "data_root",
    "data_path",
    "set_data_dir",
    "ensure_artifacts",
    "missing_artifacts",
    "build_instructions",
    "notebooks_dir",
    "__version__",
]
