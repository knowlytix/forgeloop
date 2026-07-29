"""Fetch the large trained artifacts the trilogy's notebooks load.

The small data --- the calibrated GMS stores, the fine-tuned encoders, the
calibration files, the pinned result files and the authored text inputs such as
``banking_policy.md`` --- ships inside the wheel (see :mod:`forgeloop._paths`),
so the store-building and governed-retrieval chapters run straight after
``pip install`` with no download. The large model adapters (the Qwen classifier
and drafter LoRAs, roughly 200 MB together) are not in the wheel;
:func:`ensure_artifacts` fetches only those, and only the ones still missing
under :func:`forgeloop.data_root`.

Fetch order: a local directory (``FORGELOOP_ARTIFACTS_DIR``), then a tarball URL
(``FORGELOOP_ARTIFACTS_URL``, default :data:`DEFAULT_URL` on the ForgeLoop
distribution), then a Hugging Face Hub dataset if ``FORGELOOP_ARTIFACTS_REPO``
is set. In every source the layout is one sub-directory per book, so an entry is
read from ``<book>/<entry>``.
"""

from __future__ import annotations

import os
import shutil
import tarfile
import tempfile
import urllib.request
from pathlib import Path

from forgeloop._env import getenv
from forgeloop._paths import data_root

DEFAULT_REPO = "knowlytix/forgeloop-artifacts"
# The large adapters are published as one tarball on the ForgeLoop distribution
# (linked from the docs). Override with FORGELOOP_ARTIFACTS_URL.
DEFAULT_URL = ("https://github.com/asudjianto-xml/ForgeLoop/releases/download/"
               "artifacts-v0.2.3/forgeloop-artifacts.tar.gz")

# Entries shipped inside the wheel (present under data_root without any fetch).
WHEEL_BUNDLED: tuple[str, ...] = (
    "gms_banking_store", "gms_policy_store_cap", "gms_policy_store_geode",
    "gms_regulatory_store", "gms_regulatory_cap",
    "extract_encoder_issue", "extract_encoder_product",
    "extract_geo_calibration.json", "entity_link_calibration.json",
    "capstone_run.json", "capstone_retrieval.json",
    "capstone_companions.json", "capstone_rows.json",
    "banking_policy.md", "banking_policy_full.md",
)

# Manifests per book, keyed by the marker substring of the book's directory.
BOOK_MANIFESTS: dict[str, tuple[str, ...]] = {
    "beyond-prompt-and-pray": (
        "complaint_classifier_qwen", "draft_response_lm_qwen",
        "polarity_classifier_qwen_nl", "injection_classifier_lora",
        "gms_banking_store", "gms_regulatory_store", "gms_regulatory_cap",
        "gms_policy_store_cap", "gms_policy_store_geode",
        "extract_encoder_issue", "extract_encoder_product",
        "extract_geo_calibration.json", "entity_link_calibration.json",
        "capstone_run.json", "capstone_retrieval.json",
        "capstone_companions.json", "capstone_rows.json",
    ),
    "beyond-ship-and-pray": (
        "capstone_run.json", "capstone_companions.json", "capstone_retrieval.json",
        "capstone_rows.json", "capstone_testset.json", "capstone_newprops.json",
        "capstone_trajectory_example.json",
    ),
    "beyond-chunk-and-pray": (
        "gms_annual_report_store", "enrichment", "nl2triple", "sft_supervision.json",
    ),
}

_ENV_REPO = ("FORGELOOP_ARTIFACTS_REPO", "AGENTLAB_ARTIFACTS_REPO")
_ENV_LOCAL = ("FORGELOOP_ARTIFACTS_DIR", "AGENTLAB_ARTIFACTS_DIR")
_ENV_URL = ("FORGELOOP_ARTIFACTS_URL",)


def _env(names: tuple[str, ...]) -> str | None:
    for n in names:
        v = os.environ.get(n)
        if v:
            return v
    return None


def _detect_book(root: Path | None = None) -> str | None:
    root = root or data_root()
    s = str(root)
    for book in BOOK_MANIFESTS:
        if book in s:
            return book
    # the wheel-bundled data (and its materialized cache) records its book here,
    # since the cache path does not carry the book name.
    marker = root / ".forgeloop_book"
    if marker.exists():
        name = marker.read_text().strip()
        if name in BOOK_MANIFESTS:
            return name
    return None


def _manifest(book: str | None) -> tuple[str, tuple[str, ...]]:
    book = book or _detect_book()
    if book is None:
        raise ValueError(
            f"could not infer the book from data_root ({data_root()}); pass book= "
            f"one of {list(BOOK_MANIFESTS)}")
    return book, BOOK_MANIFESTS[book]


def missing_artifacts(book: str | None = None) -> list[str]:
    """Return the manifest entries not yet present under :func:`data_root`."""
    root = data_root()
    _, entries = _manifest(book)
    return [e for e in entries if not (root / e).exists()]


def _copy_from_local(source: Path, book: str, names: list[str], root: Path) -> list[str]:
    fetched = []
    for name in names:
        # accept either <source>/<book>/<name> or <source>/<name>
        src = next((c for c in (source / book / name, source / name) if c.exists()), None)
        if src is None:
            continue
        dst = root / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dst)
        fetched.append(name)
    return fetched


def _download_from_hub(repo_id: str, book: str, names: list[str], root: Path) -> list[str]:
    from huggingface_hub import snapshot_download
    import tempfile

    patterns = []
    for name in names:
        patterns += [f"{book}/{name}", f"{book}/{name}/**"]
    with tempfile.TemporaryDirectory() as tmp:
        snapshot_download(repo_id=repo_id, repo_type="dataset",
                          local_dir=tmp, allow_patterns=patterns)
        return _copy_from_local(Path(tmp), book, names, root)


def _download_from_url(url: str, book: str, names: list[str], root: Path) -> list[str]:
    with tempfile.TemporaryDirectory() as tmp:
        tarball = Path(tmp) / "artifacts.tar.gz"
        urllib.request.urlretrieve(url, tarball)  # noqa: S310 — documented https source
        with tarfile.open(tarball) as tf:
            tf.extractall(tmp)  # noqa: S202 — trusted first-party archive
        return _copy_from_local(Path(tmp), book, names, root)


def ensure_artifacts(book: str | None = None, *, repo_id: str | None = None,
                     local_source: str | os.PathLike | None = None,
                     url: str | None = None, quiet: bool = False) -> list[str]:
    """Ensure the current book's large adapters are present under data_root.

    The small data already ships in the wheel, so this fetches only the entries
    still missing (in practice the model adapters). Returns the entries fetched
    (empty when nothing was missing). Source order: a local directory, then a
    tarball URL on the ForgeLoop distribution, then a Hugging Face Hub dataset.
    """
    root = data_root()
    book, _ = _manifest(book)
    need = missing_artifacts(book)
    if not need:
        return []
    local = local_source or _env(_ENV_LOCAL)
    repo = repo_id or _env(_ENV_REPO)
    if local:
        src = Path(local).expanduser().resolve()
        if not src.is_dir():
            raise FileNotFoundError(f"artifact source directory not found: {src}")
        if not quiet:
            print(f"fetching {len(need)} {book} artifact(s) from {src} -> {root}")
        fetched = _copy_from_local(src, book, need, root)
    elif repo:
        if not quiet:
            print(f"downloading {len(need)} {book} artifact(s) from hf://{repo} -> {root}")
        fetched = _download_from_hub(repo, book, need, root)
    else:
        src_url = url or _env(_ENV_URL) or DEFAULT_URL
        if not quiet:
            print(f"downloading {len(need)} {book} artifact(s) from {src_url} -> {root}")
        fetched = _download_from_url(src_url, book, need, root)
    still = missing_artifacts(book)
    if still and not quiet:
        print(f"warning: {len(still)} artifact(s) still missing: {', '.join(still)}")
    return fetched


def ensure_default(entry: str, root: Path | None = None) -> Path:
    """Fetch a single default artifact if absent, then return its resolved path.

    Facades call this when a caller relies on a default path so a fresh install
    fetches on first use instead of raising. ``root`` overrides the data
    directory (e.g. a facade anchored to its own book via ``book_data_root``).
    """
    root = root or data_root()
    target = root / entry
    if target.exists():
        return target
    book = _detect_book(root)
    try:
        if book is not None:
            ensure_artifacts(book)
    except Exception as e:  # noqa: BLE001 — re-raised with actionable guidance
        raise FileNotFoundError(
            f"required artifact '{entry}' is missing under {root} and could not be "
            f"fetched ({type(e).__name__}: {e}). Point FORGELOOP_ARTIFACTS_DIR at a "
            f"directory that contains it, or call forgeloop.ensure_artifacts() once the "
            f"artifact bundle has been published."
        ) from e
    if not target.exists():
        raise FileNotFoundError(
            f"artifact '{entry}' is missing under {root} and was not fetched. "
            f"Run forgeloop.ensure_artifacts() or set FORGELOOP_ARTIFACTS_DIR."
        )
    return target
