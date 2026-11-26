"""Variable-font filenames (FontCore DECISIONS.md, Area 5) are left alone by the rewriting commands, and `check` flags the old forms."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

HOUSE_FORM = [
    "Probe-Variable", "Probe-VariableUpright", "Probe-VariableItalic", "Probe-WidthVariableUpright",
    "Probe-OpticalSizeVariableItalic", "Probe-WidthVariable", "Probe-CustomVariableUpright",
    "ProbeCollection-Variable", "ProbeDisplayCondensed-Variable", "ProbeDisplayCondensed-VariableItalic",
    "ProbeBlack-WidthVariableItalic", "Probe-VariableOblique", "AugureStereoLight-WidthVariable", "RobotoFlex-CustomVariable",
    "ProbeAlt-Variable", "ParchWeight-CustomVariableUpright",
]
REWRITERS = [
    "normalize", "numbers", "compound", "abbreviations", "styles", "capitalize-after-hyphen",
    "hyphenation", "dedupe-words", "regular", "reorder-widths", "reorder-slopes", "reorder-optical",
]


def run(*args):
    return subprocess.run(
        [sys.executable, "-m", "filenametools", *map(str, args)],
        capture_output=True, text=True, cwd=ROOT, timeout=120, stdin=subprocess.DEVNULL,
    )


def listing(folder: Path):
    return sorted(p.name for p in folder.iterdir())


@pytest.mark.parametrize("command", REWRITERS)
def test_a_rewriting_command_leaves_house_form_variable_names_alone(tmp_path, command):
    for stem in HOUSE_FORM:
        (tmp_path / f"{stem}.ttf").touch()
    before = listing(tmp_path)
    flags = ["--no-preview"] if "--no-preview" in run(command, "-h").stdout else []
    result = run(command, *flags, tmp_path)
    assert result.returncode == 0, result.stdout[-400:] + result.stderr[-400:]
    assert listing(tmp_path) == before


def check_output(tmp_path, *stems):
    for stem in stems:
        (tmp_path / f"{stem}.ttf").touch()
    result = run("check", tmp_path)
    return " ".join((result.stdout + result.stderr).split())


def test_check_accepts_house_form_variable_names(tmp_path):
    assert "0 need a look" in check_output(tmp_path, *HOUSE_FORM)


@pytest.mark.parametrize(
    "stem, suggestion",
    [
        ("Probe-CondensedVariable", "ProbeCondensed-Variable"),
        ("Probe-VariableRoman", "Probe-VariableUpright"),
        ("Probe-VariableSlope", "Probe-Variable"),
        ("Probe-Variable3AxisItalic", "Probe-VariableItalic"),
        ("Probe-VariableWidth", "Probe-WidthVariable"),
        ("Probe-VariableBlackItalic", "ProbeBlack-VariableItalic"),
        ("Probe-VariableAlt", "ProbeAlt-Variable"),
        ("Probe-VF", "Probe-Variable"),
        ("Probe-Var", "Probe-Variable"),
    ],
)
def test_check_flags_old_variable_forms_and_says_what_the_house_form_is(tmp_path, stem, suggestion):
    out = check_output(tmp_path, stem)
    assert "1 need a look" in out and suggestion in out


def test_check_flags_a_weight_before_variable(tmp_path):
    out = check_output(tmp_path, "Probe-BoldVariable")
    assert "1 need a look" in out and "weight" in out.lower()


def test_check_says_the_axes_come_from_the_font(tmp_path):
    out = check_output(tmp_path, "Probe-CondensedVariable")
    assert "fontrenamer" in out


def test_compound_fixes_the_slanted_marker_without_the_old_typo(tmp_path):
    (tmp_path / "Probe-Variableslanted.ttf").touch()
    run("compound", *(["--no-preview"] if "--no-preview" in run("compound", "-h").stdout else []), tmp_path)
    assert "Probe-VariableSlanted.ttf" in listing(tmp_path) or "Probe-Variableslanted.ttf" in listing(tmp_path)
    assert not any("Varaible" in n for n in listing(tmp_path))


@pytest.mark.parametrize("workflow", ["safe", "moderate", "aggressive"])
def test_the_cleaner_leaves_house_form_variable_names_alone(workflow):
    from filenametools.FontFiles_Cleaner import process_filename

    for stem in HOUSE_FORM:
        name, changed, steps = process_filename(f"{stem}.ttf", workflow)
        assert (name, changed, steps) == (f"{stem}.ttf", False, [])
