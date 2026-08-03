"""Fail if anything secret-shaped is committed.

GitHub's own secret scanning catches vendor-issued tokens once they are pushed;
this catches them one step earlier, in a PR, and covers the two things specific
to this repo that no vendor pattern knows about: a `knowlytix` licence key, and
an absolute path off a contributor's machine.

Deliberately narrow. A scanner that cries wolf gets skipped, so every pattern
here is one where a hit is almost certainly a real problem, and the placeholder
forms the repo genuinely uses (`YOUR_KEY_HERE`, `<KNOWLYTIX_INDEX_URL>`,
`/path/to/...`) are allowed through.

    python scripts/secret_scan.py            # scan tracked files
    python scripts/secret_scan.py FILE...    # scan specific files
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# (label, pattern). Case-sensitive on purpose: the token prefixes are.
PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("OpenAI-style key", re.compile(r"\bsk-[A-Za-z0-9_-]{20,}")),
    ("Anthropic key", re.compile(r"\bsk-ant-[A-Za-z0-9_-]{20,}")),
    ("GitHub token", re.compile(r"\b(ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{30,}\b")),
    ("GitHub fine-grained token", re.compile(r"\bgithub_pat_[A-Za-z0-9_]{50,}\b")),
    ("AWS access key id", re.compile(r"\b(AKIA|ASIA)[0-9A-Z]{16}\b")),
    ("Google API key", re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b")),
    ("Hugging Face token", re.compile(r"\bhf_[A-Za-z0-9]{34,}\b")),
    ("Slack token", re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}")),
    ("private key block", re.compile(r"-----BEGIN (?:[A-Z ]+ )?PRIVATE KEY-----")),
    # A licence key is opaque, so match the assignment rather than the value.
    ("knowlytix licence key", re.compile(
        r"(?i)\b(licen[cs]e[_-]?key)\s*[:=]\s*[\"'][^\"'\s]{16,}")),
    # An absolute home directory names a real person's machine and breaks for
    # everyone else. `/path/to/...` is the sanctioned placeholder.
    ("machine-specific path", re.compile(r"(?:/Users/|/home/|C:\\\\Users\\\\)(?!\w*[<{])\w+/")),
]

# Values that look like secrets but are the documented placeholders.
ALLOWED = re.compile(
    r"YOUR_[A-Z_]*(KEY|TOKEN|SECRET)_?HERE"
    r"|<[A-Z_]+>"
    r"|/Users/(you|user|username|me)/"
    r"|/home/(you|user|username|me)/"
    r"|xxx+|placeholder|EXAMPLE|example\.com"
)

# Binaries and vendored/generated trees where a match is noise.
SKIP_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".pdf", ".ico", ".woff",
                 ".woff2", ".safetensors", ".pt", ".pth", ".bin", ".gguf"}
SKIP_PARTS = {".git", "_build", "node_modules", "__pycache__"}
# This file necessarily contains every pattern it looks for.
SELF = Path(__file__).relative_to(ROOT).as_posix()


def tracked_files() -> list[Path]:
    out = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT,
                         capture_output=True, text=True, check=True).stdout
    return [ROOT / p for p in out.split("\0") if p]


def scan(paths: list[Path]) -> list[str]:
    findings: list[str] = []
    for path in paths:
        # An explicitly-named path may sit outside the repo (a pre-commit hook
        # passes staged paths, a contributor points it at one file), so fall
        # back to the path as given rather than failing to report at all.
        rel = (path.relative_to(ROOT).as_posix()
               if path.is_absolute() and path.is_relative_to(ROOT) else str(path))
        if rel == SELF or path.suffix.lower() in SKIP_SUFFIXES:
            continue
        if SKIP_PARTS & set(path.parts):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue  # binary or unreadable: nothing to match against
        for lineno, line in enumerate(text.splitlines(), 1):
            if ALLOWED.search(line):
                continue
            for label, pattern in PATTERNS:
                m = pattern.search(line)
                if m:
                    excerpt = m.group(0)[:24]
                    findings.append(f"{rel}:{lineno}: {label}: {excerpt}...")
                    break
    return findings


def main(argv: list[str]) -> int:
    paths = [Path(a).resolve() for a in argv] if argv else tracked_files()
    findings = scan(paths)
    if not findings:
        print(f"secret scan: clean ({len(paths)} files)")
        return 0
    print(f"secret scan: {len(findings)} finding(s)\n", file=sys.stderr)
    for f in findings:
        print(f"  {f}", file=sys.stderr)
    print(
        "\nThis repo is public. If a real credential is involved, rotate it first —"
        "\nremoving the commit does not un-leak it. If it is a false positive, use a"
        "\nplaceholder (YOUR_KEY_HERE, /path/to/...) or widen ALLOWED in this script.",
        file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
