"""filenametools check: report filenames that need a look. It never renames anything."""

from __future__ import annotations

import pytest

from filenametools.FNT_Check import Records, check_stem


def codes(stem, records=None):
    return [f.code for f in check_stem(stem, records)]


@pytest.mark.parametrize("stem", ["Probe-Regular", "Probe-BoldItalic", "ProbePro-SemiCondensedLight", "FKT_Gnarly-Semibold", "Probe-Variable"])
def test_clean_names_have_no_findings(stem):
    assert codes(stem) == []


def test_no_hyphen():
    assert codes("Yearlind Bold") == ["no-hyphen"]


def test_extra_hyphens():
    assert "extra-hyphens" in codes("Probe-Sans-Bold")


@pytest.mark.parametrize("stem", ["Probe  Sans-Bold", " Probe-Bold", "Probe -Bold", "Probe- Bold", "Probe-Bold "])
def test_spaces_are_reported_with_a_suggestion(stem):
    found = [f for f in check_stem(stem) if f.code == "spaces"]
    assert found and found[0].suggestion and "  " not in found[0].suggestion


@pytest.mark.parametrize("stem", ["ProbeBold-Regular", "ProbeBold-Italic", "Probe_Light-Regular", "ProbeBlack-Bold"])
def test_a_weight_word_ending_the_family_part(stem):
    assert "weight-in-family" in codes(stem)


def test_a_weight_word_only_in_the_style_is_fine():
    assert "weight-in-family" not in codes("Probe-Italic")
    assert "weight-in-family" not in codes("Probe-Bold")


def test_a_cut_short_style_word_with_the_font_saying_the_full_word():
    records = Records(style="Condensed Bold", full="Probe Condensed Bold")
    found = [f for f in check_stem("Probe-CondBold", records) if f.code == "cut-short"]
    assert found and "Condensed" in found[0].suggestion
    assert "name record" in found[0].message


def test_a_cut_short_style_word_without_records_is_only_possible():
    found = [f for f in check_stem("Probe-Cond") if f.code == "cut-short"]
    assert found and "possible" in found[0].message and "Condensed" in found[0].suggestion


@pytest.mark.parametrize("stem", ["Probe-Carved", "Probe-Sans", "Probe-Ink", "Probe-Air", "Probe-Mono", "Probe-Display", "Probe-BoldItalic"])
def test_real_style_words_and_bespoke_words_are_not_cut_short(stem):
    assert "cut-short" not in codes(stem)


def test_a_lowercase_style():
    assert "lowercase-style" in codes("Probe-bold")


@pytest.mark.parametrize("stem", ["Probe-Bold (1)", "Probe-Bold copy", "Probe-Bold(2)"])
def test_copy_suffixes(stem):
    assert "copy-suffix" in codes(stem)


def test_the_name_table_disagrees_with_the_filename():
    records = Records(style="Light", full="Probe Light")
    found = [f for f in check_stem("Probe-Bold", records) if f.code == "name-mismatch"]
    assert found and "Bold" in found[0].message and "Light" in found[0].message


def test_the_name_table_agreeing_is_fine():
    assert codes("Probe-Bold", Records(style="Bold", full="Probe Bold")) == []


def test_regular_may_be_missing_from_the_name_table():
    assert codes("Probe-Regular", Records(style="", full="Probe")) == []


def test_the_command_lists_findings_and_renames_nothing(tmp_path):
    import subprocess
    import sys
    from pathlib import Path

    from fontTools.fontBuilder import FontBuilder

    names = ["Probe-Regular.ttf", "Probe-Cond.ttf", "Yearlind Bold.ttf"]
    for n in names:
        fb = FontBuilder(1000, isTTF=True)
        fb.setupGlyphOrder([".notdef"])
        fb.setupCharacterMap({})
        fb.setupGlyf({".notdef": __import__("fontTools.pens.ttGlyphPen", fromlist=["x"]).TTGlyphPen(None).glyph()})
        fb.setupHorizontalMetrics({".notdef": (500, 0)})
        fb.setupHorizontalHeader(ascent=800, descent=-200)
        fb.setupNameTable({"familyName": "Probe", "styleName": "Regular"})
        fb.setupOS2()
        fb.setupPost()
        fb.save(tmp_path / n)
    result = subprocess.run(
        [sys.executable, "-m", "filenametools", "check", str(tmp_path)],
        capture_output=True, text=True, timeout=120, stdin=subprocess.DEVNULL,
        cwd=Path(__file__).resolve().parents[1],
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "cut-short" in result.stdout and "no-hyphen" in result.stdout
    assert "Checked 3 files; 2 need a look. Nothing was renamed." in result.stdout
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(names)


def test_name_records_with_no_style_word_are_reported_as_odd_not_as_a_mismatch():
    assert codes("Probe-Bold", Records(style=".", full="Probe")) == ["name-records-odd"]
    assert codes("Probe-Bold", Records(style="Not for desktop use", full="")) == ["name-records-odd"]
