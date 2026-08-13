# SPDX-License-Identifier: Apache-2.0
"""Execute notebooks and verify all cells pass.

By default runs notebooks from all three books. Restrict to one or more with:

    pytest tests/test_notebooks.py --book beyond-ship-and-pray
    pytest tests/test_notebooks.py --book beyond-prompt-and-pray --book beyond-chunk-and-pray
"""
from __future__ import annotations

import pathlib

import nbformat
import pytest
from nbconvert.preprocessors import CellExecutionError, ExecutePreprocessor

_REPO = pathlib.Path(__file__).resolve().parents[1]  # forgeloop/

_BOOKS: dict[str, dict] = {
    "beyond-ship-and-pray": {
        "nb_dir": _REPO / "beyond-ship-and-pray" / "notebooks",
        "env": {"GMSTEST_REPO": str(_REPO / "beyond-ship-and-pray" / "code")},
    },
    "beyond-prompt-and-pray": {
        "nb_dir": _REPO / "beyond-prompt-and-pray" / "notebooks",
        "env": {},
    },
    "beyond-chunk-and-pray": {
        "nb_dir": _REPO / "beyond-chunk-and-pray" / "notebooks",
        "env": {},
    },
}

_TIMEOUT = 3600  # seconds per notebook (evaluation notebooks can take 15-30 min on GPU)


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    if "nb_item" not in metafunc.fixturenames:
        return
    selected = metafunc.config.getoption("--book") or list(_BOOKS)
    params, ids = [], []
    for book in selected:
        nb_dir = _BOOKS[book]["nb_dir"]
        for nb_path in sorted(nb_dir.glob("*.ipynb")):
            params.append((book, nb_path))
            ids.append(f"{book}/{nb_path.stem}")
    metafunc.parametrize("nb_item", params, ids=ids)


def test_notebook(
    nb_item: tuple, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    book, nb_path = nb_item
    monkeypatch.setenv("MPLBACKEND", "Agg")
    for k, v in _BOOKS[book]["env"].items():
        monkeypatch.setenv(k, v)

    with capsys.disabled():
        print(f"\n  executing {book}/{nb_path.name} ...", flush=True)

    nb_dir = _BOOKS[book]["nb_dir"]
    nb = nbformat.read(nb_path, as_version=4)
    ep = ExecutePreprocessor(timeout=_TIMEOUT, kernel_name="python3")
    try:
        ep.preprocess(nb, {"metadata": {"path": str(nb_dir)}})
    except CellExecutionError as exc:
        pytest.fail(str(exc))
