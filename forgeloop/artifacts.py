"""Locate the large trained artifacts the trilogy's notebooks load.

The small data --- the calibrated GMS stores, the fine-tuned encoders, the
calibration files, the pinned result files and the authored text inputs such as
``banking_policy.md`` --- ships inside the wheel (see :mod:`forgeloop._paths`),
so the store-building and governed-retrieval chapters run straight after
``pip install`` with no download. The large model adapters (the Qwen classifier
and drafter LoRAs, roughly 200 MB together) are not in the wheel and **are not
published as a download**: you train them yourself from the committed corpora
with the scripts in :data:`PRODUCERS`, which is what the Tier 2 section of the
README describes. :func:`missing_artifacts` reports which ones are absent.

:func:`ensure_artifacts` therefore copies rather than downloads by default. It
consults, in order: a local directory (``FORGELOOP_ARTIFACTS_DIR``), a tarball
URL (``FORGELOOP_ARTIFACTS_URL``), then a Hugging Face Hub dataset
(``FORGELOOP_ARTIFACTS_REPO``). None has a default value --- with none of them
set it raises and names the script that builds what is missing, rather than
reaching for a bundle that does not exist. In every source the layout is one
sub-directory per book, so an entry is read from ``<book>/<entry>``.
"""

from __future__ import annotations

import os
import shutil
import tarfile
import tempfile
import urllib.request
from pathlib import Path

from forgeloop._paths import book_data_root, data_root

# What rebuilds each artifact, as a path under its book's directory. Named in
# the error when an artifact is missing, so the message ends in a command to run
# rather than a bare path. Everything here needs the licensed substrate, and the
# Qwen-based entries want a CUDA GPU.
#
# Most stores come from the book's `00_setup.ipynb`, which runs the build
# scripts in dependency order with the right arguments -- pointing a reader at
# one script out of that sequence would have them build it wrong. The entries
# 00_setup does *not* cover name their script directly.
PRODUCERS: dict[str, str] = {
    # --- beyond-prompt-and-pray -------------------------------------------
    "gms_banking_store": "notebooks/00_setup.ipynb",
    "gms_regulatory_store": "notebooks/00_setup.ipynb",
    "gms_policy_store_cap": "notebooks/00_setup.ipynb",
    "gms_policy_store_geode": "code/scripts/build_geode_rag_store.py",
    "gms_regulatory_cap": "code/scripts/build_regulatory_cap_store.py",
    "complaint_classifier_qwen": "code/scripts/train_complaint_classifier_qwen.py",
    "draft_response_lm_qwen": "code/scripts/train_draft_response_lora.py",
    "polarity_classifier_qwen_nl": "code/scripts/train_polarity_classifier_lora.py --input nl",
    "injection_classifier_lora": "code/scripts/train_injection_classifier.py",
    "extract_encoder_issue": "code/scripts/finetune_extract_encoder.py",
    "extract_encoder_product": "code/scripts/finetune_extract_encoder.py",
    "extract_geo_calibration.json": "code/scripts/calibrate_extract_geo.py",
    "entity_link_calibration.json": "code/scripts/benchmark_entity_link.py",
    # --- beyond-ship-and-pray ---------------------------------------------
    # The pinned campaign results are committed, so these fire only if someone
    # deletes them; 00_setup re-runs the campaign that produces the set.
    "capstone_run.json": "notebooks/00_setup.ipynb",
    "capstone_rows.json": "notebooks/00_setup.ipynb",
    "capstone_companions.json": "notebooks/00_setup.ipynb",
    "capstone_retrieval.json": "notebooks/00_setup.ipynb",
    "capstone_testset.json": "notebooks/00_setup.ipynb",
    "capstone_newprops.json": "notebooks/00_setup.ipynb",
    "capstone_trajectory_example.json": "code/scripts/capture_trajectory_example.py",
    # --- beyond-chunk-and-pray --------------------------------------------
    "gms_annual_report_store": "notebooks/00_setup.ipynb",
    "enrichment": "notebooks/00_setup.ipynb",
    "nl2triple": "code/scripts/nl2triple_experiment/train_lora.py",
}

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
        # Produced by Ship-and-Pray's capstone_run.py and shipped with that
        # book, but listed here too because this book's evaluation notebooks
        # read them. _roots() below searches every book's data directory, so
        # naming them here does not make them look missing.
        "capstone_run.json", "capstone_retrieval.json",
        "capstone_companions.json", "capstone_rows.json",
    ),
    "beyond-ship-and-pray": (
        "capstone_run.json", "capstone_companions.json", "capstone_retrieval.json",
        "capstone_rows.json", "capstone_testset.json", "capstone_newprops.json",
        "capstone_trajectory_example.json",
    ),
    "beyond-chunk-and-pray": (
        # sft_supervision.json was listed here but nothing in the repo reads or
        # produces it, so it reported missing forever. Dropped.
        "gms_annual_report_store", "enrichment", "nl2triple",
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


def _roots(book: str) -> list[Path]:
    """Every data directory an artifact could legitimately live in.

    An artifact is not missing just because it sits with a different book. The
    campaign results are produced by Ship-and-Pray and read by Prompt-and-Pray's
    evaluation notebooks, so a per-book root reports false positives in whichever
    book does not own them. Checking each book's directory plus the caller's
    resolves that, and in an installed package all of these collapse to the one
    bundled ``data/``.
    """
    seen: list[Path] = []
    for cand in [book_data_root(b) for b in BOOK_MANIFESTS] + [data_root()]:
        if cand not in seen:
            seen.append(cand)
    return seen


def missing_artifacts(book: str | None = None) -> list[str]:
    """Return the manifest entries not present in any book's data directory.

    Reports only what genuinely has to be built or fetched. Previously this
    checked the caller-anchored :func:`data_root` alone, which from a checkout
    takes the editable-install fallback and lands in one book's ``code/data`` --
    so entries belonging to another book were reported missing even when they
    were on disk.
    """
    book, entries = _manifest(book)
    roots = _roots(book)
    return [e for e in entries if not any((r / e).exists() for r in roots)]


def _owner(entry: str, fallback: str | None) -> str:
    """The book whose manifest declares ``entry`` -- where its build script lives."""
    for b, entries in BOOK_MANIFESTS.items():
        if entry in entries:
            return b
    return fallback or "beyond-prompt-and-pray"


def build_instructions(book: str | None = None, entries: list[str] | None = None) -> str:
    """Return the message telling a reader how to build the missing artifacts.

    There is no published bundle to fall back on, so "missing" always means
    "not built yet". Naming the script per entry turns a dead end into the next
    command to run.
    """
    # Tolerates an undetectable book: the caller may only know the entry name,
    # and a message about a missing artifact must not itself fail on that.
    book = book or _detect_book()
    if entries is None:
        entries = missing_artifacts(book) if book else []
    if not entries:
        return f"all {book or 'known'} artifacts are present."
    lines = [
        f"{len(entries)} artifact(s) are not built yet: {', '.join(entries)}.",
        "",
        "These are built locally, not downloaded. With the knowlytix substrate",
        "installed and licensed (and a CUDA GPU for the Qwen adapters), run each",
        "of these from the repository root:",
        "",
    ]
    # An entry is built by the book that owns it, which is not always the book
    # asking for it -- the capstone results cross books. Anchor each command to
    # its owner so the path is one a reader can paste. De-duplicated because one
    # script often produces several entries (both extractor encoders, the whole
    # capstone set) and repeating it reads like extra work.
    seen: list[str] = []
    for e in entries:
        if e not in PRODUCERS:
            continue
        target = f"{_owner(e, book)}/{PRODUCERS[e]}"
        cmd = (f"    run the notebook  {target}" if target.endswith(".ipynb")
               else f"    python {target}")
        if cmd not in seen:
            seen.append(cmd)
    # 00_setup first: the scripts train on top of the stores it builds, so the
    # listed order has to be a runnable order, not manifest order.
    lines += sorted(seen, key=lambda c: not c.startswith("    run the notebook"))
    unknown = [e for e in entries if e not in PRODUCERS]
    if unknown:
        lines.append(f"    (no build step recorded for: {', '.join(unknown)})")
    lines += [
        "",
        "Already have them built elsewhere? Point FORGELOOP_ARTIFACTS_DIR at the",
        "directory holding them, or FORGELOOP_ARTIFACTS_URL / "
        "FORGELOOP_ARTIFACTS_REPO",
        "at a tarball or Hugging Face dataset you host.",
    ]
    return "\n".join(lines)


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
    import tempfile

    from huggingface_hub import snapshot_download

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

    The small data already ships in the wheel, so this covers only the entries
    still missing (in practice the model adapters). Returns the entries fetched
    (empty when nothing was missing). Source order: a local directory, then a
    tarball URL, then a Hugging Face Hub dataset -- each supplied by argument or
    environment variable. With none supplied there is nothing to fetch from, and
    it raises :func:`build_instructions` rather than guessing at a location.
    """
    book, _ = _manifest(book)
    root = book_data_root(book)          # same per-book anchoring as the check
    need = missing_artifacts(book)
    if not need:
        return []
    local = local_source or _env(_ENV_LOCAL)
    repo = repo_id or _env(_ENV_REPO)
    src_url = url or _env(_ENV_URL)
    if local:
        src = Path(local).expanduser().resolve()
        if not src.is_dir():
            raise FileNotFoundError(f"artifact source directory not found: {src}")
        if not quiet:
            print(f"fetching {len(need)} {book} artifact(s) from {src} -> {root}")
        fetched = _copy_from_local(src, book, need, root)
    elif src_url:
        if not quiet:
            print(f"downloading {len(need)} {book} artifact(s) from {src_url} -> {root}")
        fetched = _download_from_url(src_url, book, need, root)
    elif repo:
        if not quiet:
            print(f"downloading {len(need)} {book} artifact(s) from hf://{repo} -> {root}")
        fetched = _download_from_hub(repo, book, need, root)
    else:
        raise FileNotFoundError(build_instructions(book, need))
    still = missing_artifacts(book)
    if still and not quiet:
        print(f"warning: {len(still)} artifact(s) still missing: {', '.join(still)}")
    return fetched


def ensure_default(entry: str, root: Path | None = None) -> Path:
    """Resolve a single default artifact, fetching it if a source is configured.

    Facades call this when a caller relies on a default path, so a copy from a
    configured source happens on first use instead of raising. With no source
    configured the artifact simply has not been trained yet, and the error says
    which script builds it. ``root`` overrides the data directory (e.g. a facade
    anchored to its own book via ``book_data_root``).
    """
    root = root or data_root()
    target = root / entry
    if target.exists():
        return target
    book = _detect_book(root)
    try:
        if book is not None:
            ensure_artifacts(book)
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"required artifact '{entry}' is missing under {root}.\n\n{e}"
        ) from e
    except Exception as e:  # noqa: BLE001 — re-raised with actionable guidance
        raise FileNotFoundError(
            f"required artifact '{entry}' is missing under {root} and could not be "
            f"fetched ({type(e).__name__}: {e}).\n\n{build_instructions(book, [entry])}"
        ) from e
    if not target.exists():
        raise FileNotFoundError(
            f"artifact '{entry}' is missing under {root} and was not "
            f"fetched.\n\n{build_instructions(book, [entry])}"
        )
    return target
