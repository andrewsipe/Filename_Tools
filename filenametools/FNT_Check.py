#!/usr/bin/env python3
"""Report filenames that need a look. Changes nothing.

Checks a font's filename for the problems the other filenametools commands (and the name tools) trip over:
no hyphen, extra hyphens, stray spaces, a weight word stuck on the family, a cut-short style word (Cond, Bd),
a lowercase style, copy suffixes, and a filename that disagrees with the font's own name records. Where the
font's name records give the full word, that is shown as the suggestion; otherwise a dictionary guess is
marked "possible". Nothing is renamed: review the list and fix names yourself, or with the other commands.

CLI: paths... [-r] [-v] [--all]
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

# ruff: noqa: E402

import FontCore.core_console_styles as cs
from FontCore.core_font_style_dictionaries import (
    MODIFIERS,
    STYLE_WORDS,
    WEIGHT_BASES,
    WEIGHT_WORDS,
    WIDTH_BASES,
)

# Words a cut-short token can stand for (title case).
_TARGETS = sorted(
    set(WEIGHT_BASES)
    | set(WIDTH_BASES)
    | {"Regular", "Italic", "Oblique", "Roman", "Slanted", "Display", "Medium", "Semibold", "Extrabold"}
)
_KNOWN = (
    {w.lower() for w in STYLE_WORDS}
    | {w.lower() for w in WEIGHT_WORDS}
    | {w.lower() for w in WEIGHT_BASES}
    | {w.lower() for w in WIDTH_BASES}
    | {w.lower() for w in MODIFIERS}
    | {"regular", "italic", "oblique", "variable", "text", "display", "roman", "upright", "slanted"}
    | {f"{m}{w}".lower() for m in MODIFIERS for w in set(WEIGHT_BASES) | set(WIDTH_BASES)}
)


@dataclass
class Records:
    """What the font's own name records say (any may be empty)."""

    style: str = ""  # ID17, else ID2
    full: str = ""  # ID4


@dataclass
class Finding:
    code: str
    message: str
    suggestion: str = ""


def _tokens(text: str) -> List[str]:
    """Split on spaces, underscores and case or digit boundaries: 'SemiCondBold' -> Semi Cond Bold."""
    out: List[str] = []
    for part in re.split(r"[\s_]+", text):
        out += re.findall(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|\d+", part)
    return out


def _subsequence(small: str, big: str) -> bool:
    it = iter(big)
    return all(c in it for c in small)


def _candidates(token: str) -> List[str]:
    t = token.lower()
    if len(t) < 3 or t in _KNOWN:
        return []
    return [w for w in _TARGETS if len(t) < len(w) and w.lower().startswith(t[0]) and _subsequence(t, w.lower()) and (w.lower().startswith(t) or w[-1].lower() == t[-1])]


def _words(text: str) -> set:
    return {t.lower() for t in _tokens(text)}


def check_stem(stem: str, records: Optional[Records] = None) -> List[Finding]:
    """Findings for one filename stem (no extension)."""
    found: List[Finding] = []

    # copy suffixes first, so they don't confuse the other checks
    copy = re.search(r"\s*(\(\d+\)|copy( \d+)?)\s*$", stem, re.I)
    if copy:
        found.append(Finding("copy-suffix", f"ends in '{copy.group(0).strip()}' (a copied file?)", stem[: copy.start()].strip()))
        stem = stem[: copy.start()]

    tidy = re.sub(r"\s*-\s*", "-", re.sub(r"\s{2,}", " ", stem)).strip()
    if tidy != stem:
        found.append(Finding("spaces", "has stray spaces (doubled, at an end, or beside the hyphen)", tidy))
    stem = tidy

    if "-" not in stem:
        found.append(Finding("no-hyphen", "has no hyphen, so the style can't be read"))
        return found
    family, _, style = stem.partition("-")
    if "-" in style:
        found.append(Finding("extra-hyphens", "has more than one hyphen (family and style can't be told apart)"))
        style = style.replace("-", " ")

    fam_tokens = _tokens(family)
    if fam_tokens and fam_tokens[-1].lower() in _KNOWN and fam_tokens[-1].lower() in {w.lower() for w in WEIGHT_WORDS}:
        found.append(Finding("weight-in-family", f"the family part ends in a weight word ({fam_tokens[-1]})"))

    if style and style[0].islower():
        found.append(Finding("lowercase-style", "the style starts with a lowercase letter", style[0].upper() + style[1:]))

    style_tokens = _tokens(style)
    record_words = _words(" ".join(filter(None, [records.style, records.full])) if records else "")
    for tok in style_tokens:
        low = tok.lower()
        if low in _KNOWN or len(low) < 3:
            continue
        from_record = [w for w in (records and _tokens(records.style + " " + records.full) or []) if len(w) > len(low) and w.lower().startswith(low[0]) and _subsequence(low, w.lower())]
        if from_record:
            found.append(Finding("cut-short", f"'{tok}' looks cut short; the font's name record says {from_record[0]}", from_record[0]))
            continue
        guesses = _candidates(tok)
        if guesses:
            found.append(Finding("cut-short", f"'{tok}' looks cut short (possible: {', '.join(guesses)})", guesses[0]))

    if records and (records.style or records.full):
        # Compare with spaces, case and punctuation removed (Semibold = SemiBold = Semi Bold). Variable and the
        # Upright/Roman pairing words are filename conventions the name records don't carry.
        def squash(text: str) -> str:
            return re.sub(r"[^a-z0-9]", "", text.lower())

        pool = squash(f"{records.style} {records.full}")
        record_known = any(t.lower() in _KNOWN for t in _tokens(f"{records.style} {records.full}"))
        skip = {"regular", "variable", "upright", "roman"}
        missing = [t for t in style_tokens if t.lower() not in skip and squash(t) not in pool and not _candidates(t) and t.lower() in _KNOWN]
        if missing and not record_known:
            found.append(
                Finding(
                    "name-records-odd",
                    f"the font's name records hold no style word (they say '{records.style or records.full}')",
                )
            )
        elif missing:
            found.append(
                Finding(
                    "name-mismatch",
                    f"the filename style says {' '.join(style_tokens)} but the font's name records say {records.style or records.full}",
                )
            )
    return found


def _records_for(path: Path) -> Optional[Records]:
    try:
        from fontTools.ttLib import TTFont

        font = TTFont(path, lazy=True, recalcTimestamp=False)
        try:
            name = font["name"]

            def get(i: int) -> str:
                r = name.getName(i, 3, 1, 0x409)
                return r.toUnicode() if r else ""

            return Records(style=get(17) or get(2), full=get(4))
        finally:
            font.close()
    except Exception:
        return None


def _plural(n: int, noun: str) -> str:
    return f"{n} {noun}{'' if n == 1 else 's'}"


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="filenametools check",
        description="List font filenames that need a look. Changes nothing.",
    )
    p.add_argument("paths", nargs="+", help="font files or folders")
    p.add_argument("-r", "--recursive", action="store_true", help="include subfolders")
    p.add_argument("-v", "--verbose", action="store_true", help="list every file for every finding")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(list(sys.argv[1:] if argv is None else argv))
    from FontCore.core_file_collector import collect_font_files

    files: List[Path] = [Path(f) for f in collect_font_files(args.paths, recursive=args.recursive)]
    if not files:
        print("No font files found.")
        return 0

    by_code: dict = {}
    flagged = 0
    for path in files:
        findings = check_stem(path.stem, _records_for(path))
        if findings:
            flagged += 1
        for f in findings:
            by_code.setdefault(f.code, []).append((path, f))

    for code, items in sorted(by_code.items(), key=lambda kv: -len(kv[1])):
        print(f"\n{code}: {_plural(len(items), 'file')}")
        for path, f in (items if args.verbose else items[:5]):
            tail = f"  ->  {f.suggestion}" if f.suggestion else ""
            print(f"  {path.name}: {f.message}{tail}")
        if not args.verbose and len(items) > 5:
            print(f"  ... {len(items) - 5} more (use -v to list them all)")
    print(f"\nChecked {_plural(len(files), 'file')}; {flagged} need a look. Nothing was renamed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
