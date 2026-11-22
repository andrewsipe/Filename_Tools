"""One vocabulary: the tools read their term lists from FontCore, so a word added there reaches every tool."""

import importlib
import subprocess
import sys
from pathlib import Path

import pytest
from FontCore import core_font_style_dictionaries as d

ROOT = Path(__file__).resolve().parents[1]


def mod(name):
    return importlib.import_module(f"filenametools.{name}")


def test_slope_terms_come_from_fontcore():
    slopes = mod("FNT_ReorderSlopes")
    assert slopes.BASE_SLOPE_TERMS == set(d.SLOPE_BASES)
    assert {"Cursive", "Kursiv"} <= slopes.BASE_SLOPE_TERMS
    assert slopes.SLOPE_MODIFIERS == set(d.MODIFIERS)


def test_width_terms_come_from_fontcore():
    widths = mod("FNT_ReorderWidths")
    assert widths.BASE_WIDTH_TERMS == set(d.WIDTH_BASES)
    assert widths.WIDTH_MODIFIERS == set(d.MODIFIERS)


def test_hyphen_terms_come_from_fontcore():
    for name in ("FNT_HyphenationConsistency", "FNT_StyleNameExpander"):
        m = mod(name)
        assert m.HYPHEN_LEFT_TERMS == set(d.HYPHEN_LEFT_TERMS), name
        assert m.HYPHEN_RIGHT_TERMS == set(d.HYPHEN_RIGHT_TERMS), name
        assert m.COMPOUND_PREFIXES == set(d.COMPOUND_PREFIXES), name


def test_optical_terms_start_from_fontcore_and_add_filename_only_extras():
    optical = mod("FNT_ReorderOpticalSizes")
    assert set(d.OPTICAL_BASES) <= optical.OPTICAL_SIZE_TERMS
    assert {"Titling", "Fine", "Small"} <= optical.OPTICAL_SIZE_TERMS  # the tool's own extras are kept


def test_the_unused_stat_lists_are_gone():
    assert not hasattr(d, "ELIDABLE_WEIGHT_NAMES")
    assert not hasattr(d, "ELIDABLE_WIDTH_NAMES")


@pytest.mark.parametrize("stem, expected", [("Probe-CursiveBold", "Probe-BoldCursive"), ("Probe-KursivBold", "Probe-BoldKursiv"), ("Probe-ObliqueBold", "Probe-BoldOblique")])
def test_reorder_slopes_moves_cursive_and_kursiv_like_any_slope(tmp_path, stem, expected):
    (tmp_path / f"{stem}.ttf").touch()
    result = subprocess.run(
        [sys.executable, "-m", "filenametools", "reorder-slopes", "--no-preview", str(tmp_path)],
        capture_output=True, text=True, cwd=ROOT, timeout=120, stdin=subprocess.DEVNULL,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert [p.name for p in tmp_path.iterdir()] == [f"{expected}.ttf"]


def _regular(tmp_path, stem):
    (tmp_path / f"{stem}.otf").touch()
    result = subprocess.run(
        [sys.executable, "-m", "filenametools", "regular", str(tmp_path)],
        capture_output=True, text=True, cwd=ROOT, timeout=120, stdin=subprocess.DEVNULL,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return [p.stem for p in tmp_path.iterdir()][0]


@pytest.mark.parametrize(
    "stem, expected",
    [
        ("Probe-Italic", "Probe-RegularItalic"),
        ("Probe-Oblique", "Probe-RegularOblique"),
        ("Probe-Cursive", "Probe-RegularCursive"),
        ("Probe-Slanted", "Probe-RegularSlanted"),
        ("Probe-Backslant", "Probe-RegularBackslant"),
        ("Probe-Kursiv", "Probe-RegularKursiv"),
        ("Probe-Condensed", "Probe-CondensedRegular"),
        ("Probe-Compact", "Probe-CompactRegular"),
        ("Probe-Display", "Probe-Display"),  # ambiguous: left for review
        ("Probe-Carved", "Probe-Carved"),
        ("Probe-Bold", "Probe-Bold"),
        ("Probe-Roman", "Probe-Roman"),
    ],
)
def test_regular_uses_the_shared_slope_and_width_words(tmp_path, stem, expected):
    assert _regular(tmp_path, stem) == expected
