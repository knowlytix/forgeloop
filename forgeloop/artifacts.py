"""Fetch the trained artifacts the trilogy's notebooks load.

``forgeloop`` ships code; the trained artifacts each book's notebooks read
(classifier and drafter adapters, calibrated GMS stores, fine-tuned encoders,
and pinned result files) are large and are not committed or bundled in the
wheel. :func:`ensure_artifacts` fetches only what is missing under
:func:`forgeloop.data_root`, from a local directory
(``FORGELOOP_ARTIFACTS_DIR``) or the Hugging Face Hub dataset repo named by
``FORGELOOP_ARTIFACTS_REPO`` (default :data:`DEFAULT_REPO`).

Each book has its own manifest; the book is inferred from the resolved data
directory. On the Hub the bundle is laid out one sub-directory per book, so an
entry is fetched from ``<book>/<entry>``.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from forgeloop._env import getenv
from forgeloop._paths import data_root

DEFAULT_REPO = "knowlytix/forgeloop-artifacts"

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


def _env(names: tuple[str, ...]) -> str | None:
    for n in names:
        v = os.environ.get(n)
        if v:
            return v
    return None


def _detect_book(root: Path | None = None) -> str | None:
    s = str(root or data_root())
    for book in BOOK_MANIFESTS:
        if book in s:
            return book
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


def ensure_artifacts(book: str | None = None, *, repo_id: str | None = None,
                     local_source: str | os.PathLike | None = None,
                     quiet: bool = False) -> list[str]:
    """Ensure the current book's trained artifacts are present under data_root.

    Returns the entries fetched (empty when all were already present).
    """
    root = data_root()
    book, _ = _manifest(book)
    need = missing_artifacts(book)
    if not need:
        return []
    local = local_source or _env(_ENV_LOCAL)
    if local:
        src = Path(local).expanduser().resolve()
        if not src.is_dir():
            raise FileNotFoundError(f"artifact source directory not found: {src}")
        if not quiet:
            print(f"fetching {len(need)} {book} artifact(s) from {src} -> {root}")
        fetched = _copy_from_local(src, book, need, root)
    else:
        repo = repo_id or _env(_ENV_REPO) or DEFAULT_REPO
        if not quiet:
            print(f"downloading {len(need)} {book} artifact(s) from hf://{repo} -> {root}")
        fetched = _download_from_hub(repo, book, need, root)
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
