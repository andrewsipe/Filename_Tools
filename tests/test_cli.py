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


def test_corpus_with_no_discoverable_terms_does_not_crash(tmp_path):
    """Single-word filenames yield no terms; the summary used to divide by zero."""
    (tmp_path / "Kappa.ttf").write_bytes(b"")
    (tmp_path / "Lambda.otf").write_bytes(b"")
    result = _cli("corpus", str(tmp_path))
    assert result.returncode == 0, result.stderr[-400:]
    assert "ZeroDivisionError" not in result.stderr
    assert (tmp_path / "corpus_analysis_report_simple.md").exists()


# ---- grouped help ---------------------------------------------------------------------------------------

def test_every_target_is_in_exactly_one_group():
    from filenametools.cli import GROUPS

    listed = [t for _, targets in GROUPS for t in targets]
    assert sorted(listed) == sorted(set(TARGETS) | {"reorder"})
    assert len(listed) == len(set(listed))


def test_the_help_shows_the_groups_in_order_with_every_target():
    from filenametools.cli import GROUPS

    out = _cli("-h").stdout
    positions = [out.index(title) for title, _ in GROUPS]
    assert positions == sorted(positions)
    for target in TARGETS:
        assert f"  {target} " in out or f"  {target}\n" in out


def test_the_look_first_group_comes_before_anything_that_renames():
    from filenametools.cli import GROUPS

    titles = [t for t, _ in GROUPS]
    assert titles[0].lower().startswith("look")
    assert set(GROUPS[0][1]) == {"check", "patterns", "corpus"}


def test_the_help_says_which_targets_change_nothing_and_suggests_an_order():
    out = _cli("-h").stdout
    assert "change nothing" in out and "Suggested order" in out


def test_every_target_still_runs_under_its_old_name():
    for target in TARGETS:
        assert _cli(target, "-h").returncode == 0, target


@pytest.mark.parametrize("target", ["patterns", "corpus"])
def test_help_for_the_targets_that_read_their_own_arguments(target):
    result = _cli(target, "-h")
    assert result.returncode == 0 and f"usage: filenametools {target}" in result.stdout
    assert "No such file" not in result.stdout + result.stderr


# ---- one reorder command ---------------------------------------------------------------------------------

def test_reorder_runs_the_same_code_as_the_old_names(tmp_path):
    for n in ("Probe-CondensedBold.otf", "Probe-BoldCondensed.otf"):
        (tmp_path / n).touch()
    old = _cli("reorder-widths", "-n", "--no-preview", str(tmp_path)).stdout
    new = _cli("reorder", "widths", "-n", "--no-preview", str(tmp_path)).stdout
    assert old == new and "→" in new


@pytest.mark.parametrize("family", ["widths", "slopes", "optical", "replace"])
def test_reorder_help_for_each_family(family):
    result = _cli("reorder", family, "-h")
    assert result.returncode == 0 and "usage:" in result.stdout


def test_reorder_without_a_family_lists_them():
    result = _cli("reorder")
    assert result.returncode == 2
    text = result.stdout + result.stderr
    assert all(f in text for f in ("widths", "slopes", "optical", "replace"))


def test_reorder_with_an_unknown_family_says_so():
    result = _cli("reorder", "colors")
    assert result.returncode == 2 and "colors" in result.stdout + result.stderr


def test_the_old_names_still_work():
    for t in ("reorder-widths", "reorder-slopes", "reorder-optical", "reorder-replace"):
        assert _cli(t, "-h").returncode == 0


def test_the_help_lists_reorder_first_in_its_group():
    out = _cli("-h").stdout
    assert out.index("  reorder ") < out.index("reorder-widths")
