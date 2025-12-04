#!/usr/bin/env python3
"""
Expand common font style abbreviations and normalize compound style words in filenames.

Rules (exactly as provided; no extras):
- Abbreviations → Expansions
  Bk→Book, Blk→Black, Reg→Regular, Bd→Bold, Lt→Light, XLt→Extralight, XBd→Extrabold,
  Med→Medium, SmBd→Semibold, Rg→Regular, Th→Thin, Md→Medium, Ita→Italic, Obl→Oblique,
  Cd→Condensed, Cond→Condensed, ExtLt→Extralight, ExtBd→Extrabold, UltLt→Ultralight,
  UltBd→Ultrabold, SemBd→Semibold, SemiCond→SemiCondensed, UltraCond→UltraCondensed,
  ExtraCond→ExtraCondensed, Nar→Narrow, Nr→Narrow, Obliq→Oblique, Exp→Expanded

- Compound Normalizations
  SemiBold→Semibold, ExtraBold→Extrabold, UltraBold→Ultrabold,
  SemiLight→Semilight, ExtraLight→Extralight, UltraLight→Ultralight,
  ExtraBlack→Extrablack, UltraBlack→Ultrablack,
  ExtraThin→Extrathin, UltraThin→Ultrathin,
  ExtraHeavy→Extraheavy, UltraHeavy→Ultraheavy

Safety against over-expansion (e.g., "Med" in "Medium" → "Mediumium"):
- Only expand when the abbreviation does not directly prefix the full expansion segment.
- If a token already starts with the expansion text (case-insensitive), skip the abbreviation.

Targets filenames only (stems by default; extension unchanged). Supports single file(s), directory, and recursive scanning. Includes conflict handling, dry-run, verbose, and optional interactive confirmation when multiple changes are possible.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# Optional: fontTools for --var mode (name table normalization on variable fonts)
try:  # noqa: SIM105
    from fontTools.ttLib import TTFont  # type: ignore

    _FONTTOOLS_AVAILABLE = True
except Exception:  # pragma: no cover - env dependent
    TTFont = None  # type: ignore
    _FONTTOOLS_AVAILABLE = False

import FontCore.core_console_styles as cs


# --- Rules (only what the user specified) -------------------------------------------------------

ABBREV_RULES: Dict[str, str] = {
    "Blk": "Black",
    "Bd": "Bold",
    "Bk": "Book",
    "Cd": "Condensed",
    "Cn": "Condensed",
    "Cnd": "Condensed",
    "Cond": "Condensed",
    "DBb": "Demibold",
    "DmBd": "Demibold",
    "DemBd": "Demibold",
    "DmBold": "Demibold",
    "DemBold": "Demibold",
    "Exp": "Expanded",
    "XCn": "ExtraCondensed",
    "XCnd": "ExtraCondensed",
    "XCond": "ExtraCondensed",
    "ExCn": "ExtraCondensed",
    "ExCnd": "ExtraCondensed",
    "ExCond": "ExtraCondensed",
    "ExtraCond": "ExtraCondensed",
    "XBd": "Extrabold",
    "XBold": "Extrabold",
    "ExBd": "Extrabold",
    "ExBold": "Extrabold",
    "XLt": "Extralight",
    "ExLt": "Extralight",
    "XLight": "Extralight",
    "ExLight": "Extralight",
    "It": "Italic",
    "Ita": "Italic",
    "italic": "Italic",
    "Lt": "Light",
    "Md": "Medium",
    "Med": "Medium",
    "Nr": "Narrow",
    "Nar": "Narrow",
    "Obl": "Oblique",
    "Obliq": "Oblique",
    "Rg": "Regular",
    "Reg": "Regular",
    "Rnd": "Round",
    "SmBd": "Semibold",
    "SemBd": "Semibold",
    "SemiBd": "Semibold",
    "SmBold": "Semibold",
    "SemBold": "Semibold",
    "SmLt": "Semilight",
    "SemLt": "Semilight",
    "SemiLt": "Semilight",
    "SmLight": "Semilight",
    "SemLight": "Semilight",
    "SmCd": "SemiCondensed",
    "SmCnd": "SemiCondensed",
    "SmCond": "SemiCondensed",
    "SemiCd": "SemiCondensed",
    "SemiCnd": "SemiCondensed",
    "SemiCond": "SemiCondensed",
    "Ult": "Ultra",
    "UltBd": "Ultrabold",
    "UltBold": "Ultrabold",
    "UltLt": "Ultralight",
    "UltLight": "Ultralight",
    "UltraCond": "UltraCondensed",
    "GX": "Variable",
    "VF": "Variable",
    "Var": "Variable",
    "Wd": "Wide",
    "Wid": "Wide",
    "SC": "Smallcaps",
}

COMPOUND_NORMALIZATIONS: Dict[str, str] = {
    "VariableRegular-Variable": "Variable",
    "VariableItalic-Variable": "VariableItalic",
    "VariableOblique-Variable": "VariableOblique",
    "Variable-Variable": "Variable",
    "Variable-Italic": "VariableItalic",
    "Variable-Oblique": "VariableOblique",
    "VariableVariable": "Variable",
    "Small-Caps": "Smallcaps",
    "SmallCaps": "Smallcaps",
    "SemiBold": "Semibold",
    "DemiBold": "Demibold",
    "ExtraBold": "Extrabold",
    "UltraBold": "Ultrabold",
    "SemiLight": "Semilight",
    "ExtraLight": "Extralight",
    "UltraLight": "Ultralight",
    "ExtraBlack": "Extrablack",
    "UltraBlack": "Ultrablack",
    "ExtraThin": "Extrathin",
    "UltraThin": "Ultrathin",
    "ExtraHeavy": "Extraheavy",
    "UltraHeavy": "Ultraheavy",
    "XBold": "Extrabold",
    "XLight": "Extralight",
    "XThin": "Extrathin",
    "XBlack": "Extrablack",
    "XHeavy": "Extraheavy",
    "Roman": "Regular",
    "Round-ed": "Rounded",
    "Slant-ed": "Slanted",
    "Semi-Mono": "SemiMono",
    "Semi Mono": "SemiMono",
}


# Partition rules into truncated vs non-truncated (computed at load and after user updates)
NON_TRUNCATED_RULES: Dict[str, str] = {}
TRUNCATED_RULES: Dict[str, str] = {}

# Hyphen placement preferences (seed with examples requested by user)
HYPHEN_LEFT_TERMS: set[str] = {
    # Widths (hyphen left Ex -Condensed)
    "Condensed",
    "SemiCondensed",
    "ExtraCondensed",
    "UltraCondensed",
    "Compressed",
    "SemiCompressed",
    "ExtraCompressed",
    "UltraCompressed",
    "Compact",
    "SemiCompact",
    "ExtraCompact",
    "UltraCompact",
    "Narrow",
    "SemiNarrow",
    "ExtraNarrow",
    "UltraNarrow",
    "Expanded",
    "SemiExpanded",
    "ExtraExpanded",
    "UltraExpanded",
    "Extended",
    "SemiExtended",
    "ExtraExtended",
    "UltraExtended",
    "Wide",
    "SemiWide",
    "ExtraWide",
    "UltraWide",
    "Variable",
}
HYPHEN_RIGHT_TERMS: set[str] = {
    # Non-WWS (hyphen right Ex Display-)
    "Display",
    "Text",
    "Caption",
    "Subhead",
    "Headline",
    "Title",
    "Poster",
    "Deck",
    "Micro",
    "Round",
    "Mono",
    "Sans",
    "Slab",
    "Serif",
}


# Known style prefixes that should not keep a hyphen before the next style token.
# Example fixes: "Semi-Condensed" -> "SemiCondensed", "Extra-Light" -> "ExtraLight".
COMPOUND_PREFIXES: set[str] = {
    "Semi",
    "Extra",
    "Demi",
    "X",
    "Ultra",
    "Super",
}


def recompute_rule_partitions() -> None:
    global NON_TRUNCATED_RULES, TRUNCATED_RULES
    non_trunc: Dict[str, str] = {}
    trunc: Dict[str, str] = {}
    for abbr, exp in ABBREV_RULES.items():
        try:
            if exp.lower().startswith(abbr.lower()) and len(abbr) < len(exp):
                trunc[abbr] = exp
            else:
                non_trunc[abbr] = exp
        except Exception:
            non_trunc[abbr] = exp
    NON_TRUNCATED_RULES = non_trunc
    TRUNCATED_RULES = trunc


# Initialize partitions at import time
recompute_rule_partitions()


# --- Helpers ------------------------------------------------------------------------------------


def split_stem_and_suffixes(filename: str) -> Tuple[str, str]:
    path = Path(filename)
    suffixes = "".join(path.suffixes)
    if not suffixes:
        return filename, ""
    stem = path.name[: -len(suffixes)]
    return stem, suffixes


def ensure_unique_destination(path: Path) -> Path:
    if not path.exists():
        return path
    counter = 1
    stem, suffixes = split_stem_and_suffixes(path.name)
    candidate = path
    while candidate.exists():
        candidate = path.with_name(f"{stem} ({counter}){suffixes}")
        counter += 1
    return candidate


def tokenize_style_segments(basename: str) -> List[str]:
    """Tokenize a basename into candidate word-like segments while preserving delimiters.

    We want to expand only style-like tokens, but filenames can have delimiters like
    '-', '_', ' ', '.', etc. We split on non-alphabetic boundaries while keeping them
    in the output so reconstruction is lossless.
    """
    if not basename:
        return []
    parts: List[str] = []
    buf: List[str] = []
    is_alpha_prev = basename[0].isalpha()
    for ch in basename:
        is_alpha = ch.isalpha()
        if is_alpha == is_alpha_prev:
            buf.append(ch)
        else:
            parts.append("".join(buf))
            buf = [ch]
            is_alpha_prev = is_alpha
    if buf:
        parts.append("".join(buf))
    return parts


def _is_alpha_part(text: str) -> bool:
    return bool(text) and text[0].isalpha()


def _lookahead_matches_remainder(
    parts: List[str],
    index: int,
    token: str,
    abbrev_low: str,
    expansion_low: str,
) -> bool:
    """Return True if characters following the abbrev appear to already complete the expansion.

    This checks within the current token's remainder and then subsequent alphabetic parts,
    ignoring delimiters, to see if they form the full remainder of the expansion. Used only
    when the abbreviation is a true prefix of the expansion (truncated form).
    """
    remainder_needed = expansion_low[len(abbrev_low) :]
    if not remainder_needed:
        return False

    token_low = token.lower()
    # remainder present within current token after the abbrev slice
    in_token_remainder = token_low[len(abbrev_low) :]

    collected = in_token_remainder
    j = index + 1
    while len(collected) < len(remainder_needed) and j < len(parts):
        nxt = parts[j]
        if _is_alpha_part(nxt):
            collected += nxt.lower()
        j += 1

    if len(collected) < len(remainder_needed):
        return False

    return collected[: len(remainder_needed)] == remainder_needed


def safe_expand_abbrev(
    parts: List[str],
    index: int,
    token: str,
    abbrev: str,
    expansion: str,
) -> Optional[str]:
    """Return expanded token or None if we should skip to avoid double-expansion.

    Case-insensitive match for the provided abbreviation, but the expansion text is
    canonical-cased as specified in the rules. Safety: if the token already starts with
    the full expansion (case-insensitive), skip to avoid producing things like Mediumium.
    """
    if not token or not abbrev:
        return None

    token_low = token.lower()
    abbrev_low = abbrev.lower()
    expansion_low = expansion.lower()

    # Safety: if token already starts with expansion (case-insensitive), skip
    if token_low.startswith(expansion_low):
        return None

    # Only allow expansion when token is exactly the abbrev, or when the remainder
    # begins with an uppercase letter (CamelCase boundary), to avoid false positives like 'Regal'.
    if token_low == abbrev_low:
        # For truncated forms (e.g., Med→Medium), look ahead across delimiters to avoid Mediumium
        if expansion_low.startswith(abbrev_low) and _lookahead_matches_remainder(
            parts, index, token, abbrev_low, expansion_low
        ):
            return None
        return expansion

    if token_low.startswith(abbrev_low):
        remainder_current = token[len(abbrev) :]
        if not remainder_current or remainder_current[0].isupper():
            # Truncated-form safety across delimiters
            if expansion_low.startswith(abbrev_low) and _lookahead_matches_remainder(
                parts, index, token, abbrev_low, expansion_low
            ):
                return None
            return f"{expansion}{remainder_current}"

    # Suffix expansion at CamelCase boundary
    # Case A: truncated forms (e.g., ExtralightIt -> ExtralightItalic)
    if (
        expansion_low.startswith(abbrev_low)
        and len(abbrev) < len(expansion)
        and token_low.endswith(abbrev_low)
    ):
        pos = len(token) - len(abbrev)
        if pos >= 1 and token[pos].isupper():
            return f"{token[:pos]}{expansion}"

    # Case B: general non-truncated suffix (e.g., LemonSansRnd -> LemonSansRound)
    if token_low.endswith(abbrev_low):
        pos = len(token) - len(abbrev)
        if pos >= 1 and token[pos].isupper():
            return f"{token[:pos]}{expansion}"

    # Case C: infix at CamelCase boundaries (e.g., ...SemBdItalic -> ...SemiboldItalic)
    # Match exact-cased abbrev inside token where it's between CamelCase boundaries
    pos_infix = token.find(abbrev)
    if pos_infix > 0:
        end = pos_infix + len(abbrev)
        # left boundary: preceding char is lower, current is upper (CamelCase split)
        left_boundary = token[pos_infix - 1].islower() and token[pos_infix].isupper()
        # right boundary: next char is upper or end-of-token
        right_boundary = (end == len(token)) or token[end].isupper()
        if left_boundary and right_boundary:
            return f"{token[:pos_infix]}{expansion}{token[end:]}"

    return None


def apply_abbreviation_rules_to_basename(
    basename: str, rules: Dict[str, str]
) -> Tuple[str, List[Tuple[str, str]]]:
    """Expand abbreviations within alphabetic segments while preserving delimiters.

    Returns (new_basename, changes) where changes is a list of (old, new) fragments applied.
    """
    parts = tokenize_style_segments(basename)
    if not parts:
        return basename, []

    changes: List[Tuple[str, str]] = []
    new_parts: List[str] = []

    for idx, part in enumerate(parts):
        if not _is_alpha_part(part):
            new_parts.append(part)
            continue

        # Try each abbrev; prefer longer keys first to reduce partial overshadowing
        replaced = False
        for abbrev in sorted(rules.keys(), key=len, reverse=True):
            expansion = rules[abbrev]
            candidate = safe_expand_abbrev(parts, idx, part, abbrev, expansion)
            if candidate is not None and candidate != part:
                changes.append((part, candidate))
                new_parts.append(candidate)
                replaced = True
                break
        if not replaced:
            new_parts.append(part)

    new_basename = "".join(new_parts)
    return new_basename, changes


def apply_compound_normalizations(text: str) -> Tuple[str, List[Tuple[str, str]]]:
    """Apply case-sensitive compound normalizations anywhere in the basename.

    Returns (new_text, changes) with unique, non-overlapping replacements.
    """
    if not text:
        return text, []

    # To avoid cascading overlap, perform sequential non-overlapping replaces preferring longer keys
    changes: List[Tuple[str, str]] = []
    result = text
    for src, dst in sorted(
        COMPOUND_NORMALIZATIONS.items(), key=lambda kv: len(kv[0]), reverse=True
    ):
        if src in result and dst not in result:
            new_result = result.replace(src, dst)
            if new_result != result:
                changes.append((src, dst))
                result = new_result
        else:
            # If dst already present, still attempt replacement to normalize mixed forms
            new_result = result.replace(src, dst)
            if new_result != result:
                changes.append((src, dst))
                result = new_result
    return result, changes


def apply_hyphen_placement_rules(text: str) -> Tuple[str, List[Tuple[str, str]]]:
    """Normalize hyphen placement around known style terms.

    - For left-hyphen terms (e.g., Condensed): enforce "-Condensed" (prefix hyphen) when adjacent.
    - For right-hyphen terms (e.g., Display): enforce "Display-" (suffix hyphen) when adjacent.

    We only adjust within the stem portion; caller should pass stem, not full filename.
    """
    if not text:
        return text, []

    changes: List[Tuple[str, str]] = []
    # original retained implicitly via `changes` tuples

    # Remove duplicate hyphens like "-Term-" artifacts are left to other passes; here we only move/delete one side.

    # Left-hyphen terms: ensure a hyphen before the term and no hyphen after it in mid-phrase cases.
    for term in sorted(HYPHEN_LEFT_TERMS, key=len, reverse=True):
        # Pattern scenarios:
        # 1) FooTerm-Bar -> Foo-TermBar
        # 2) Foo-Term-Bar -> Foo-TermBar (remove hyphen after)
        # 3) FooTermBar -> unchanged (we don't insert hyphen unless immediately adjacent to a prior alpha chunk)
        # We'll handle 1 & 2 by replacing "Term-" with "-Term"
        before = text
        text = text.replace(f"{term}-", f"-{term}")
        if text != before:
            changes.append((before, text))

        # If missing left hyphen but preceded by alpha and not already hyphenated, add hyphen before term
        # Scan occurrences
        i = 0
        while True:
            idx = text.find(term, i)
            if idx == -1:
                break
            left_ok = idx > 0 and text[idx - 1] == "-"
            # Preceded by alpha and not already left-hyphenated
            if not left_ok and idx > 0 and text[idx - 1].isalpha():
                # Insert a hyphen before term; if there's a hyphen after the term, leave it (another rule will clean)
                new_text = text[:idx] + "-" + text[idx:]
                changes.append((text, new_text))
                text = new_text
                i = idx + len(term) + 1
            else:
                i = idx + len(term)

        # Remove hyphen after left-hyphen term when we have "-Term-" (avoid double hyphens)
        before = text
        text = text.replace(f"-{term}-", f"-{term}")
        if text != before:
            changes.append((before, text))

    # Right-hyphen terms: ensure a hyphen after the term and no hyphen before it in mid-phrase cases.
    for term in sorted(HYPHEN_RIGHT_TERMS, key=len, reverse=True):
        # Replace "-Term" (left hyphen) with "Term-"
        before = text
        text = text.replace(f"-{term}", f"{term}-")
        if text != before:
            changes.append((before, text))

        # If missing right hyphen but followed by alpha and not already hyphenated, add hyphen after term
        i = 0
        while True:
            idx = text.find(term, i)
            if idx == -1:
                break
            end = idx + len(term)
            right_ok = end < len(text) and text[end] == "-"
            if not right_ok and end < len(text) and text[end].isalpha():
                new_text = text[:end] + "-" + text[end:]
                changes.append((text, new_text))
                text = new_text
                i = end + 1
            else:
                i = end

        # Remove hyphen before right-hyphen term when we have "-Term-" (avoid double hyphens)
        before = text
        text = text.replace(f"-{term}-", f"{term}-")
        if text != before:
            changes.append((before, text))

    # Collapse any leftover double (or more) hyphens to a single hyphen
    collapsed = re.sub(r"-{2,}", "-", text)
    if collapsed != text:
        changes.append((text, collapsed))
        text = collapsed

    return text, changes


def apply_compound_prefix_unhyphen(text: str) -> Tuple[str, List[Tuple[str, str]]]:
    """Remove hyphen immediately after known style prefixes when followed by a letter.

    This turns "Semi-Bold" into "SemiBold", "Extra-LightItalic" into "ExtraLightItalic".
    Case-sensitive; only removes the hyphen if the next character is alphabetic, to avoid
    catching non-style contexts.
    """
    if not text:
        return text, []
    changes: List[Tuple[str, str]] = []
    try:
        if not COMPOUND_PREFIXES:
            return text, changes
        # Build a regex like: (Semi|Extra|Ultra|Super)-(?=[A-Za-z])
        alt = "|".join(sorted(COMPOUND_PREFIXES, key=len, reverse=True))
        pattern = re.compile(rf"(?P<prefix>{alt})-(?=[A-Za-z])")

        def _repl(m: re.Match[str]) -> str:
            return m.group("prefix")

        before = text
        after = pattern.sub(_repl, text)
        if after != before:
            changes.append((before, after))
            text = after
    except Exception:
        return text, changes
    return text, changes


def build_new_filename(original_name: str) -> Tuple[str, List[Tuple[str, str]]]:
    stem, suffixes = split_stem_and_suffixes(original_name)
    # Pass 1: non-truncated abbreviations
    after_nontrunc, changes1 = apply_abbreviation_rules_to_basename(
        stem, NON_TRUNCATED_RULES
    )
    # Pass 2: truncated abbreviations (with look-ahead safety baked in)
    after_trunc, changes2 = apply_abbreviation_rules_to_basename(
        after_nontrunc, TRUNCATED_RULES
    )
    # Pass 3: hyphen placement normalization (on stem)
    after_hyphen, changes3 = apply_hyphen_placement_rules(after_trunc)
    # Pass 4: fix split compound prefixes (Semi-*, Extra-*, etc.)
    after_unhyphen, changes4 = apply_compound_prefix_unhyphen(after_hyphen)
    # Pass 5: compound normalization
    after_comp, changes5 = apply_compound_normalizations(after_unhyphen)
    new_name = f"{after_comp}{suffixes}"
    return new_name, (changes1 + changes2 + changes3 + changes4 + changes5)


@dataclass
class RenameDecision:
    old_name: str
    new_name: str
    changes: List[Tuple[str, str]]
    destination: Path


def compute_rename(file_path: Path) -> RenameDecision:
    old_name = file_path.name
    new_name, changes = build_new_filename(old_name)
    destination = file_path.with_name(new_name)
    return RenameDecision(
        old_name=old_name, new_name=new_name, changes=changes, destination=destination
    )


def present_decision_interactive(decision: RenameDecision) -> Optional[str]:
    """If multiple distinct changes or potentially ambiguous, present choices.

    Currently, we consider ambiguity when there is more than one change fragment.
    Choices:
      1) Apply all (default)
      2) Apply only compound normalizations
      3) Apply only abbreviation expansions
      4) Skip this file
    Returns chosen new_name, or None to skip.
    """
    old_name = decision.old_name
    stem, suffixes = split_stem_and_suffixes(old_name)

    # Build variant names using the current pipeline ordering
    stem_abbrev_non, _ = apply_abbreviation_rules_to_basename(stem, NON_TRUNCATED_RULES)
    stem_abbrev_all, _ = apply_abbreviation_rules_to_basename(
        stem_abbrev_non, TRUNCATED_RULES
    )
    stem_hyphen, _ = apply_hyphen_placement_rules(stem_abbrev_all)
    stem_all, _ = apply_compound_normalizations(stem_hyphen)

    # Alternatives: compound-only and abbrev-only
    stem_comp_only, _ = apply_compound_normalizations(stem)
    stem_abbrev_only, _ = apply_abbreviation_rules_to_basename(
        stem, {**NON_TRUNCATED_RULES, **TRUNCATED_RULES}
    )

    option_all = f"{stem_all}{suffixes}"
    option_comp = f"{stem_comp_only}{suffixes}"
    option_abbr = f"{stem_abbrev_only}{suffixes}"

    cs.StatusIndicator("info").add_message(
        f"Ambiguous changes for {cs.fmt_file(old_name)}:"
    ).emit()
    cs.StatusIndicator("info").add_item(
        f"1) {cs.fmt_change(old_name, option_all)}  (all changes)"
    ).emit()
    cs.StatusIndicator("info").add_item(
        f"2) {cs.fmt_change(old_name, option_comp)}  (compound only)"
    ).emit()
    cs.StatusIndicator("info").add_item(
        f"3) {cs.fmt_change(old_name, option_abbr)}  (abbrev only)"
    ).emit()
    cs.StatusIndicator("info").add_item("4) Skip").emit()

    while True:
        try:
            choice = input("Choose [1-4] (blank = 1): ").strip()
        except EOFError:
            choice = ""
        if choice == "" or choice == "1":
            return option_all
        if choice == "2":
            return option_comp
        if choice == "3":
            return option_abbr
        if choice == "4":
            return None


def iter_target_files(root: Path, recursive: bool) -> Iterable[Path]:
    def _is_hidden(p: Path) -> bool:
        try:
            parts = p.relative_to(root).parts
        except Exception:
            parts = p.parts
        return any(seg.startswith(".") for seg in parts)

    if root.is_file():
        if not _is_hidden(root):
            yield root
        return
    if root.is_dir():
        if recursive:
            for p in root.rglob("*"):
                if p.is_file() and not _is_hidden(p):
                    yield p
        else:
            for p in root.iterdir():
                if p.is_file() and not _is_hidden(p):
                    yield p


def _is_supported_font_file(path: Path) -> bool:
    try:
        return path.suffix.lower() in {".ttf", ".otf"}
    except Exception:
        return False


def _normalize_name_table_compounds(font: "TTFont") -> int:
    """Apply COMPOUND_NORMALIZATIONS to all name strings; return number of records updated."""
    if "name" not in font:
        return 0
    name_table = font["name"]
    changed = 0
    for rec in list(name_table.names):
        try:
            try:
                old_text = rec.toUnicode()
            except Exception:
                old_text = str(getattr(rec, "string", ""))
            new_text, _changes = apply_compound_normalizations(old_text)
            if new_text != old_text:
                name_table.setName(
                    new_text, rec.nameID, rec.platformID, rec.platEncID, rec.langID
                )
                changed += 1
        except Exception:
            continue
    return changed


def perform_var_name_normalization(
    file_path: Path,
    *,
    dry_run: bool,
    verbose: bool,
) -> Tuple[bool, Optional[str]]:
    if not _FONTTOOLS_AVAILABLE:
        return False, "fontTools not available"
    try:
        font = TTFont(str(file_path))  # type: ignore
    except Exception as exc:  # noqa: BLE001
        return False, f"open failed: {exc}"

    try:
        if "fvar" not in font:
            if verbose:
                cs.StatusIndicator("unchanged").add_file(
                    str(file_path)
                ).with_explanation("skip non-variable").emit()
            try:
                font.close()
            except Exception:
                pass
            return False, None

        updated = _normalize_name_table_compounds(font)
        if updated == 0:
            if verbose:
                cs.StatusIndicator("unchanged").add_file(str(file_path)).emit()
            try:
                font.close()
            except Exception:
                pass
            return False, None

        if dry_run:
            cs.StatusIndicator("info", dry_run=True).add_file(
                str(file_path)
            ).add_message("DRY-RUN name table").add_field(
                "updated_records", updated
            ).emit()
            try:
                font.close()
            except Exception:
                pass
            return True, None

        # Save in place
        try:
            font.save(str(file_path))
        finally:
            try:
                font.close()
            except Exception:
                pass

        cs.StatusIndicator("updated").add_file(str(file_path)).add_message(
            "name table"
        ).add_field("updated_records", updated).emit()
        return True, None
    except Exception as exc:  # noqa: BLE001
        try:
            font.close()
        except Exception:
            pass
        return False, f"normalize failed: {exc}"


_DUP_SUFFIX_RE = __import__("re").compile(r"^(.+?) \((\d+)\)(\..+)?$")


def cleanup_numbered_duplicates(
    root: Path, recursive: bool, *, dry_run: bool, verbose: bool
) -> None:
    """Remove ' (n)' suffixes when the clean name is available.

    Mirrors Files_CapitalizeAfterHyphen's behavior for consistency.
    """
    for file_path in iter_target_files(root, recursive):
        m = _DUP_SUFFIX_RE.match(file_path.name)
        if not m:
            continue
        base = m.group(1)
        ext = m.group(3) or ""
        clean_name = f"{base}{ext}"
        clean_path = file_path.with_name(clean_name)
        if not clean_path.exists():
            if dry_run:
                cs.StatusIndicator("info", dry_run=True).add_message(
                    f"cleanup in {cs.fmt_file(str(file_path.parent))}"
                ).add_values(old_value=file_path.name, new_value=clean_name).emit()
            else:
                try:
                    file_path.rename(clean_path)
                    cs.StatusIndicator("updated").add_message(
                        f"cleanup in {cs.fmt_file(str(file_path.parent))}"
                    ).add_values(old_value=file_path.name, new_value=clean_name).emit()
                except Exception as exc:  # noqa: BLE001
                    cs.StatusIndicator("error").add_file(
                        str(file_path)
                    ).with_explanation(f"cleanup failed: {exc}").emit()
        elif verbose:
            cs.StatusIndicator("unchanged").add_file(file_path.name).with_explanation(
                "cleanup skip (exists)"
            ).emit()


def perform_rename(
    file_path: Path,
    *,
    interactive: bool,
    dry_run: bool,
    conflict: str,
    verbose: bool,
) -> Tuple[bool, Optional[str]]:
    """Execute rename and return (changed, error_message)."""
    decision = compute_rename(file_path)

    if decision.new_name == decision.old_name:
        if verbose:
            cs.StatusIndicator("unchanged").add_file(str(file_path)).emit()
        return False, None

    target_name = decision.new_name

    if interactive and len(decision.changes) > 1:
        choice = present_decision_interactive(decision)
        if choice is None:
            if verbose:
                cs.StatusIndicator("unchanged").add_file(
                    str(file_path)
                ).with_explanation("skipped").emit()
            return False, None
        target_name = choice

    destination = file_path.with_name(target_name)
    if destination.exists():
        if conflict == "skip":
            if verbose:
                cs.StatusIndicator("warning").add_file(target_name).with_explanation(
                    "exists, skipping"
                ).emit()
            return False, None
        if conflict == "unique":
            destination = ensure_unique_destination(destination)
        # overwrite: proceed

    if dry_run:
        cs.StatusIndicator("info", dry_run=True).add_message(
            f"DRY-RUN rename in {cs.fmt_file(str(file_path.parent))}"
        ).add_values(old_value=decision.old_name, new_value=destination.name).emit()
        return True, None

    try:
        file_path.rename(destination)
        cs.StatusIndicator("updated").add_message(
            f"in {cs.fmt_file(str(file_path.parent))}"
        ).add_values(old_value=decision.old_name, new_value=destination.name).emit()
        return True, None
    except Exception as exc:  # noqa: BLE001
        return False, f"rename failed for {file_path}: {exc}"


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Expand common font style abbreviations and normalize compound style words in filenames."
        )
    )
    parser.add_argument(
        "paths",
        nargs="+",
        help="File(s) and/or director(y/ies) to process",
    )
    parser.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        help="Recurse into directories (files only)",
    )
    parser.add_argument(
        "-n",
        "--dry-run",
        action="store_true",
        help="Show what would change without renaming",
    )
    parser.add_argument(
        "--conflict",
        choices=("skip", "unique", "overwrite"),
        default="unique",
        help="On name conflict: skip, create a unique name, or overwrite (default: unique)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Show unchanged files and additional info",
    )
    parser.add_argument(
        "-i",
        "--interactive",
        action="store_true",
        help="Prompt when multiple changes are possible (decision menu)",
    )
    parser.add_argument(
        "--var",
        dest="var_mode",
        action="store_true",
        help="Apply compound normalizations to the name table of variable .ttf/.otf (no file renaming)",
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        default=True,
        help="Remove ' (n)' suffixes when the clean name is available (default: enabled)",
    )
    parser.add_argument(
        "--no-cleanup",
        action="store_false",
        dest="cleanup",
        help="Skip the cleanup pass (keeps ' (n)' suffixes)",
    )
    # Future extension points (explicitly allowed by user): rules additions
    parser.add_argument(
        "--rules-file",
        dest="rules_file",
        help="JSON file with {abbr: expansion} and {compound: normalized} overrides/additions",
    )
    parser.add_argument(
        "--add-rule",
        action="append",
        default=[],
        metavar="ABBR=EXPANSION",
        help="Add or override a single abbreviation rule (can repeat)",
    )
    parser.add_argument(
        "--add-compound",
        action="append",
        default=[],
        metavar="FROM=TO",
        help="Add or override a single compound normalization rule (can repeat)",
    )
    parser.add_argument(
        "--add-hyphen-left",
        action="append",
        default=[],
        metavar="TERM",
        help="Add a term that prefers a hyphen on the left (e.g., -Condensed)",
    )
    parser.add_argument(
        "--add-hyphen-right",
        action="append",
        default=[],
        metavar="TERM",
        help="Add a term that prefers a hyphen on the right (e.g., Display-)",
    )
    return parser.parse_args(argv)


def load_user_rules(args: argparse.Namespace) -> None:
    # Only apply user-provided additions; do not auto-extend beyond the provided lists.
    # --add-rule ABBR=EXPANSION
    for raw in args.add_rule or []:
        if "=" not in raw:
            cs.StatusIndicator("warning").with_explanation(
                f"invalid --add-rule (expected ABBR=EXPANSION): {raw}"
            ).emit()
            continue
        abbr, exp = raw.split("=", 1)
        abbr = abbr.strip()
        exp = exp.strip()
        if not abbr or not exp:
            cs.StatusIndicator("warning").with_explanation(
                f"invalid --add-rule (empty key/value): {raw}"
            ).emit()
            continue
        ABBREV_RULES[abbr] = exp

    # --add-compound FROM=TO
    for raw in args.add_compound or []:
        if "=" not in raw:
            cs.StatusIndicator("warning").with_explanation(
                f"invalid --add-compound (expected FROM=TO): {raw}"
            ).emit()
            continue
        src, dst = raw.split("=", 1)
        src = src.strip()
        dst = dst.strip()
        if not src or not dst:
            cs.StatusIndicator("warning").with_explanation(
                f"invalid --add-compound (empty key/value): {raw}"
            ).emit()
            continue
        COMPOUND_NORMALIZATIONS[src] = dst

    # --rules-file JSON with optional keys: abbreviations, compounds
    if args.rules_file:
        import json

        try:
            with open(args.rules_file, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception as exc:  # noqa: BLE001
            cs.StatusIndicator("error").with_explanation(
                f"failed to read rules file: {exc}"
            ).emit()
            return

        abbrs = data.get("abbreviations") or data.get("abbr") or {}
        if isinstance(abbrs, dict):
            for k, v in abbrs.items():
                if isinstance(k, str) and isinstance(v, str):
                    ABBREV_RULES[k] = v
        comps = data.get("compounds") or data.get("compound") or {}
        if isinstance(comps, dict):
            for k, v in comps.items():
                if isinstance(k, str) and isinstance(v, str):
                    COMPOUND_NORMALIZATIONS[k] = v
        hy_left = data.get("hyphen_left") or data.get("hyphen-left") or []
        if isinstance(hy_left, list):
            for term in hy_left:
                if isinstance(term, str) and term:
                    HYPHEN_LEFT_TERMS.add(term)
        hy_right = data.get("hyphen_right") or data.get("hyphen-right") or []
        if isinstance(hy_right, list):
            for term in hy_right:
                if isinstance(term, str) and term:
                    HYPHEN_RIGHT_TERMS.add(term)

    # Inline additions for hyphen placement
    for term in args.add_hyphen_left or []:
        t = term.strip()
        if t:
            HYPHEN_LEFT_TERMS.add(t)
    for term in args.add_hyphen_right or []:
        t = term.strip()
        if t:
            HYPHEN_RIGHT_TERMS.add(t)
    # Whenever rules change, recompute the partitions
    recompute_rule_partitions()


def main(argv: Sequence[str]) -> int:
    args = parse_args(argv)

    # Load user-provided rule additions (explicit opt-in)
    load_user_rules(args)

    total_files = 0
    changed = 0
    errors = 0

    for raw_path in args.paths:
        path = Path(raw_path)
        if not path.exists():
            cs.StatusIndicator("warning").add_file(raw_path).with_explanation(
                "path not found"
            ).emit()
            continue

        for file_path in iter_target_files(path, args.recursive):
            # --var mode: operate on name table of variable fonts only
            if args.var_mode:
                if not _is_supported_font_file(file_path):
                    continue
                total_files += 1
                did_change, error_message = perform_var_name_normalization(
                    file_path,
                    dry_run=args.dry_run,
                    verbose=args.verbose,
                )
                if did_change:
                    changed += 1
                if error_message is not None:
                    errors += 1
                    cs.StatusIndicator("error").add_file(
                        str(file_path)
                    ).with_explanation(error_message).emit()
                continue

            # Default: filename renamer
            total_files += 1
            did_change, error_message = perform_rename(
                file_path,
                interactive=args.interactive,
                dry_run=args.dry_run,
                conflict=args.conflict,
                verbose=args.verbose,
            )
            if did_change:
                changed += 1
            if error_message is not None:
                errors += 1
                cs.StatusIndicator("error").with_explanation(error_message).emit()

    # Cleanup pass applies only to filename renamer mode
    if not args.var_mode and args.cleanup:
        for raw_path in args.paths:
            path = Path(raw_path)
            if path.exists():
                cleanup_numbered_duplicates(
                    path, args.recursive, dry_run=args.dry_run, verbose=args.verbose
                )

    # Summary line
    cs.fmt_processing_summary(
        dry_run=args.dry_run,
        updated=changed,
        unchanged=total_files - changed,
        errors=errors,
    )
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
