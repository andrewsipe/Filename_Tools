"""`filenametools compound` writes the house form: one-word weights (Semibold), joined widths (SemiCondensed, XXCondensed)."""

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def run(tmp_path, stem):
    (tmp_path / f"{stem}.otf").touch()
    result = subprocess.run(
        [sys.executable, "-m", "filenametools", "compound", str(tmp_path)],
        capture_output=True, text=True, cwd=ROOT, timeout=120, stdin=subprocess.DEVNULL,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return [p.stem for p in tmp_path.iterdir()][0]


@pytest.mark.parametrize(
    "stem, expected",
    [
        ("Probe-SemiBold", "Probe-Semibold"),
        ("Probe-Semi Bold", "Probe-Semibold"),
        ("Probe-ExtraLight", "Probe-Extralight"),
        ("Probe-Extra Light", "Probe-Extralight"),
        ("Probe-SemiCondensed", "Probe-SemiCondensed"),
        ("Probe-Semi Condensed", "Probe-SemiCondensed"),
        ("Probe-XXCondensed", "Probe-XXCondensed"),
        ("Probe-XX Condensed", "Probe-XXCondensed"),
        ("Probe-Semi Mono", "Probe-SemiMono"),
        ("Probe-Small Caps", "Probe-Smallcaps"),
        ("Probe-SemiBoldItalic", "Probe-SemiboldItalic"),
        ("Probe-Regular", "Probe-Regular"),
    ],
)
def test_the_house_form(tmp_path, stem, expected):
    assert run(tmp_path, stem) == expected


def test_a_family_name_is_left_alone(tmp_path):
    assert run(tmp_path, "Semi Bold Sans-Regular") == "Semi Bold Sans-Regular"


def test_the_help_describes_the_house_form():
    out = subprocess.run([sys.executable, "-m", "filenametools", "-h"], capture_output=True, text=True, cwd=ROOT).stdout
    assert "SemiBold -> Semi Bold" not in out and "Semibold" in out
