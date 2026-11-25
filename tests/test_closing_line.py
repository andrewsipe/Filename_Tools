"""Every renaming command ends with the same one-sentence summary."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _cli(*args) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "filenametools", *map(str, args)],
        capture_output=True, text=True, cwd=ROOT, timeout=120, stdin=subprocess.DEVNULL,
    )


def _text(r) -> str:
    return " ".join((r.stdout + r.stderr).split())


def test_cleaner_ends_with_the_sentence(tmp_path):
    (tmp_path / "Probe_Bold.ttf").touch()
    first = _cli("cleaner", "--workflow", "light", tmp_path)
    assert first.returncode == 0 and "Updated 1, unchanged 0, no problems." in _text(first)
    second = _cli("cleaner", "--workflow", "light", tmp_path)
    assert second.returncode == 0 and "Updated 0, unchanged 1, no problems." in _text(second)


def test_cleaner_preview_says_would_update(tmp_path):
    (tmp_path / "Probe_Bold.ttf").touch()
    r = _cli("cleaner", "--workflow", "light", "-n", tmp_path)
    assert "Would update 1, unchanged 0, no problems." in _text(r)
