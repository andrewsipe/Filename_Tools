#!/usr/bin/env python3
"""
Normalize compound style words in filenames (stem only) and optionally in
variable font name tables.

Focus: Compound normalization only.

CLI:
- paths... [-r] [-n] [--conflict skip|unique|overwrite] [-v] [--cleanup|--no-cleanup]
- Optional: --var to normalize name table compounds in variable .ttf/.otf (no renames)
- Optional dictionary tuning: --rules-file, --add-compound FROM=TO
"""

from __future__ import annotations

import argparse
import sys
import re
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple, Optional

import core.core_console_styles as cs


# Optional: fontTools for --var mode
try:  # noqa: SIM105
    from fontTools.ttLib import TTFont  # type: ignore

    _FONTTOOLS_AVAILABLE = True
except Exception:  # pragma: no cover - env dependent
    TTFont = None  # type: ignore
    _FONTTOOLS_AVAILABLE = False

# Modifiers that trigger the normalization pattern
MODIFIERS = ["Semi", "Demi", "Extra", "Ultra", "Super", "X"]

# Characters that should remain capitalized after modifiers (widths/italics)
CAPITALIZE_CHECK = {"C", "E", "N", "W", "I"}

# Width suffixes/words that should be followed by capitalized weights
WIDTH_PATTERNS = ["-", "nded", "ensed", "ssed", "Wide", "Narrow", "Compact"]

# Exceptional terms that pattern recognition miss or incorrectly case set
COMPOUND_NORMALIZATIONS: Dict[str, str] = {
    "italic": "Italic",
    "oblique": "Oblique",
    "slanted": "Slanted",
    "VF": "Variable",
    "Variableitalic": "VariableItalic",
    "Variableoblique": "VariableOblique",
    "Variableslanted": "VaraibleSlanted",
    "VariableRegular-Variable": "-Variable",
    "VariableItalic-Variable": "-VariableItalic",
    "VariableOblique-Variable": "-VariableOblique",
    "VariableSlanted-Variable": "-VariableSlanted",
    "Variable-Variable": "-Variable",
    "Variable-Italic": "-VariableItalic",
    "Variable-Oblique": "-VariableOblique",
    "Variable-Slanted": "-VariableSlanted",
    "VariableVariable": "-Variable",
    "Small-Caps": "Smallcaps",
    "SmallCaps": "Smallcaps",
    "ItalicSmallCaps": "SmallcapsItalic",
    "ObliqueSmallCaps": "SmallcapsOblique",
    "Roman": "Regular",
    "Round-ed": "Rounded",
    "Slant-ed": "Slanted",
    "Semimono": "SemiMono",
    "Semi-Mono": "SemiMono",
    "Semi Mono": "SemiMono",
    "Xtall": "XTall",
    "--": "-",
}


def normalize_modifier_compounds(text: str) -> Tuple[str, List[Tuple[str, str]]]:
    """
    Normalize compound words with modifiers (Semi, Demi, Extra, Ultra, X).

    Rules:
    - Weights after modifiers: lowercase (SemiBold -> Semibold)
    - Widths after modifiers: capitalize (Semicondensed -> SemiCondensed)
    - Capitalize after C, E, N, W, I; lowercase everything else

    Returns:
        Tuple of (normalized_text, list of changes)
    """
    if not text:
        return text, []

    changes: List[Tuple[str, str]] = []
    result = text

    # Sort modifiers by length (longest first) to handle "Extra" before "X"
    for modifier in sorted(MODIFIERS, key=len, reverse=True):
        # Find all occurrences of modifier followed by a capital letter
        pattern = f"{modifier}([A-Z][a-z]*)"

        def replace_match(match: re.Match) -> str:
            full_match = match.group(0)
            following_word = match.group(1)
            first_char = following_word[0]

            # Check if first character should remain capitalized
            if first_char in CAPITALIZE_CHECK:
                # Keep capitalized (e.g., SemiCondensed stays SemiCondensed)
                return full_match
            else:
                # Lowercase the following word (e.g., SemiBold -> Semibold)
                normalized = f"{modifier}{following_word.lower()}"
                if normalized != full_match:
                    changes.append((full_match, normalized))
                return normalized

        result = re.sub(pattern, replace_match, result)

    return result, changes


def normalize_width_weight_compounds(text: str) -> Tuple[str, List[Tuple[str, str]]]:
    """
    Normalize weights that follow width terms.

    Fixes cases like: Condensedthin -> CondensedThin, Widelight -> WideLight

    Rules:
    - After width patterns (nded, ensed, Wide, Narrow, Compact), capitalize following weight

    Returns:
        Tuple of (normalized_text, list of changes)
    """
    if not text:
        return text, []

    changes: List[Tuple[str, str]] = []
    result = text

    # Sort patterns by length (longest first) to handle overlaps
    for width_pattern in sorted(WIDTH_PATTERNS, key=len, reverse=True):
        # Find width pattern followed by a lowercase letter (uncapitalized weight)
        pattern = f"({width_pattern})([a-z][a-z]*)"

        def replace_match(match: re.Match) -> str:
            full_match = match.group(0)
            width_part = match.group(1)
            weight_part = match.group(2)

            # Capitalize the first letter of the weight
            normalized = f"{width_part}{weight_part.capitalize()}"
            if normalized != full_match:
                changes.append((full_match, normalized))
            return normalized

        result = re.sub(pattern, replace_match, result)

    return result, changes


def apply_compound_normalizations_with_pattern(
    text: str, dictionary: dict[str, str]
) -> Tuple[str, List[Tuple[str, str]]]:
    """
    Apply pattern-based normalization first, then dictionary exceptions.

    Order of operations:
    1. Normalize modifiers (Semi, Demi, Extra, Ultra, X)
    2. Normalize width+weight combinations
    3. Apply dictionary for exceptions

    This allows the dictionary to override pattern results for special cases.
    """
    if not text:
        return text, []

    all_changes: List[Tuple[str, str]] = []

    # Step 1: Apply modifier-based normalization
    result, modifier_changes = normalize_modifier_compounds(text)
    all_changes.extend(modifier_changes)

    # Step 2: Apply width+weight normalization
    result, width_changes = normalize_width_weight_compounds(result)
    all_changes.extend(width_changes)

    # Step 3: Apply dictionary for exceptions (sorted by length, longest first)
    for src, dst in sorted(dictionary.items(), key=lambda kv: len(kv[0]), reverse=True):
        if src in result:
            new_result = result.replace(src, dst)
            if new_result != result:
                all_changes.append((src, dst))
                result = new_result

    return result, all_changes


def apply_compound_normalizations(text: str) -> Tuple[str, List[Tuple[str, str]]]:
    if not text:
        return text, []
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
            new_result = result.replace(src, dst)
            if new_result != result:
                changes.append((src, dst))
                result = new_result
    return result, changes


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


def build_new_filename(original_name: str) -> Tuple[str, List[Tuple[str, str]]]:
    stem, suffixes = split_stem_and_suffixes(original_name)
    new_stem, changes = apply_compound_normalizations_with_pattern(
        stem, COMPOUND_NORMALIZATIONS
    )
    return f"{new_stem}{suffixes}", changes


def _is_supported_font_file(path: Path) -> bool:
    try:
        return path.suffix.lower() in {".ttf", ".otf"}
    except Exception:
        return False


def _normalize_name_table_compounds(font: "TTFont") -> int:
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
    dry_run: bool,
    conflict: str,
    verbose: bool,
) -> Tuple[bool, str | None]:
    original_value = file_path.name
    new_value, changes = build_new_filename(original_value)
    if new_value == original_value:
        if verbose:
            cs.StatusIndicator("unchanged").add_file(str(file_path)).emit()
        return False, None

    destination = file_path.with_name(new_value)
    if destination.exists():
        if conflict == "skip":
            if verbose:
                cs.StatusIndicator("warning").add_file(new_value).with_explanation(
                    "exists, skipping"
                ).emit()
            return False, None
        if conflict == "unique":
            destination = ensure_unique_destination(destination)
        # overwrite: proceed

    if dry_run:
        cs.StatusIndicator("info", dry_run=True).add_message(
            f"DRY-RUN rename in {cs.fmt_file(str(file_path.parent))}"
        ).add_values(old_value=original_value, new_value=destination.name).emit()
        return True, None

    try:
        file_path.rename(destination)
        cs.StatusIndicator("updated").add_message(
            f"in {cs.fmt_file(str(file_path.parent))}"
        ).add_values(old_value=original_value, new_value=destination.name).emit()
        return True, None
    except Exception as exc:  # noqa: BLE001
        return False, f"rename failed for {file_path}: {exc}"


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Normalize compound style words in filenames, optionally in variable font name tables."
        ),
    )
    parser.add_argument(
        "paths", nargs="+", help="File(s) and/or director(y/ies) to process"
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
        help="On name conflict: skip, unique, or overwrite (default: unique)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Show unchanged files and additional info",
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
    parser.add_argument(
        "--rules-file",
        dest="rules_file",
        help="JSON file with {compounds} overrides/additions",
    )
    parser.add_argument(
        "--add-compound",
        action="append",
        default=[],
        metavar="FROM=TO",
        help="Add or override a single compound normalization rule (can repeat)",
    )
    return parser.parse_args(argv)


def load_user_rules(args: argparse.Namespace) -> None:
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

    if args.rules_file:
        import json

        try:
            with open(args.rules_file, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception as exc:  # noqa: BLE001
            cs.StatusIndicator("error").with_explanation(
                f"failed to read rules file: {exc}"
            ).emit()
            data = {}
        comps = data.get("compounds") or data.get("compound") or {}
        if isinstance(comps, dict):
            for k, v in comps.items():
                if isinstance(k, str) and isinstance(v, str):
                    COMPOUND_NORMALIZATIONS[k] = v


def main(argv: Sequence[str]) -> int:
    args = parse_args(argv)
    load_user_rules(args)
    total_files = 0
    changed = 0
    errors = 0

    if args.var_mode:
        for raw_path in args.paths:
            path = Path(raw_path)
            if not path.exists():
                cs.StatusIndicator("warning").add_file(raw_path).with_explanation(
                    "path not found"
                ).emit()
                continue
            for file_path in iter_target_files(path, args.recursive):
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
    else:
        for raw_path in args.paths:
            path = Path(raw_path)
            if not path.exists():
                cs.StatusIndicator("warning").add_file(raw_path).with_explanation(
                    "path not found"
                ).emit()
                continue
            for file_path in iter_target_files(path, args.recursive):
                total_files += 1
                did_change, error_message = perform_rename(
                    file_path,
                    dry_run=args.dry_run,
                    conflict=args.conflict,
                    verbose=args.verbose,
                )
                if did_change:
                    changed += 1
                if error_message is not None:
                    errors += 1
                    cs.StatusIndicator("error").with_explanation(error_message).emit()

        if args.cleanup:
            for raw_path in args.paths:
                path = Path(raw_path)
                if path.exists():
                    cleanup_numbered_duplicates(
                        path, args.recursive, dry_run=args.dry_run, verbose=args.verbose
                    )

    cs.fmt_processing_summary(
        dry_run=args.dry_run,
        updated=changed,
        unchanged=total_files - changed,
        errors=errors,
    )
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
