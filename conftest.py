# SPDX-License-Identifier: Apache-2.0
"""Shared pytest configuration for cross-book tests."""
from __future__ import annotations


def pytest_addoption(parser: object) -> None:
    parser.addoption(
        "--book",
        action="append",
        default=None,
        metavar="BOOK",
        help=(
            "Limit notebook tests to this book directory "
            "(may be repeated; default: all three). "
            "Choices: beyond-ship-and-pray, beyond-prompt-and-pray, beyond-chunk-and-pray."
        ),
    )
