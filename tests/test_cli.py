"""The `filenametools` entry point dispatches to the original scripts."""

from __future__ import annotations

import importlib
import inspect
import subprocess
import sys
from pathlib import Path

import pytest

from filenametools import __version__
from filenametools.cli import TARGETS

ROOT = Path(__file__).resolve().parents[1]


def _cli(*args: str, cwd: Path = ROOT) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "filenametools", *args],
        capture_output=True,
        text=True,
        cwd=cwd,
        timeout=120,
        stdin=subprocess.DEVNULL,
    )


@pytest.mark.parametrize("target", sorted(TARGETS))
def test_every_target_imports_and_has_main(target):
    module = importlib.import_module(f"filenametools.{TARGETS[target][0]}")
    assert callable(module.main)
    assert len(inspect.signature(module.main).parameters) <= 1


def test_every_script_file_has_a_target():
    scripts = {p.stem for p in (ROOT / "filenametools").glob("F*.py")}
    assert scripts == {module for module, _ in TARGETS.values()}


def test_version_and_target_list():
    assert __version__ in _cli("--version").stdout
    listing = _cli().stdout
    for name in TARGETS:
        assert name in listing


def test_unknown_target_fails():
    result = _cli("nope")
    assert result.returncode == 2
    assert "unknown target" in result.stderr


@pytest.mark.parametrize("target", ["normalize", "reorder-widths", "cleaner"])
def test_help_names_the_command(target):
    out = _cli(target, "-h").stdout
    assert f"filenametools {target}" in out


def test_normalize_renames_files(tmp_path):
    (tmp_path / "Font Two-Regular.ttf").write_bytes(b"")
    result = _cli("normalize", "--no-preview", str(tmp_path))
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "FontTwo-Regular.ttf").exists()
    assert not (tmp_path / "Font Two-Regular.ttf").exists()


def test_dry_run_changes_nothing(tmp_path):
    (tmp_path / "Probe Wide-Light.otf").write_bytes(b"")
    result = _cli("reorder-widths", "-n", str(tmp_path))
    assert result.returncode == 0, result.stderr
    assert [p.name for p in tmp_path.iterdir()] == ["Probe Wide-Light.otf"]
