#!/usr/bin/env python3
"""
Insert "Regular" into font filenames missing an explicit weight designation.

MINIMALIST APPROACH - Conservative by design:
- Small, focused term lists
- Skips ambiguous cases
- Requires human audit for edge cases

Examples that WILL be changed:
  Font.otf → Font-Regular.otf
  Font-Italic.otf → Font-RegularItalic.otf
  Font-Condensed.otf → Font-CondensedRegular.otf

Examples that will be SKIPPED (require manual review):
  Font-Display.otf → unchanged (ambiguous)
  Font-Text.otf → unchanged (ambiguous)
  Font-Poster.otf → unchanged (ambiguous)
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence, Set, Tuple

# Add project root to path for FontCore imports (works for root and subdirectory scripts)
# ruff: noqa: E402
_project_root = Path(__file__).parent
while (
    not (_project_root / "FontCore").exists() and _project_root.parent != _project_root
):
    _project_root = _project_root.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import FontCore.core_console_styles as cs


# --- Minimal Term Dictionaries ------------------------------------------------------------------

# Core weight terms (modifiers like Semi/Demi/Extra/Ultra are ignored)
# If any of these appear in the filename, skip it
CORE_WEIGHT_TERMS: Set[str] = {
    "Thin",
    "Hairline",
    "Light",
    "Regular",
    "Normal",
    "Roman",
    "Book",
    "Medium",
    "Bold",
    "Black",
    "Heavy",
    "Fat",
}

# We don't need to list SemiBold, ExtraBold, etc.
# If "Bold" is in the name, that's enough to skip

# Only the most common slope terms
SLOPE_TERMS: Set[str] = {
    "Italic",
    "Oblique",
    "Slant",
    "Cursive",
    "Smallcap",
}

# Only the most common width terms (core terms, modifiers handled automatically)
WIDTH_TERMS: Set[str] = {
    "Condensed",
    "Compressed",
    "Narrow",
    "Extended",
    "Expanded",
    "Expand",
    "Wide",
}


# --- Helpers ------------------------------------------------------------------------------------


def split_stem_and_suffixes(filename: str) -> Tuple[str, str]:
    """
    Split filename into stem and file extension.
    Uses the last period to determine the extension.
    """
    if "." not in filename:
        return filename, ""

    # Split on the last period only
    stem, ext = filename.rsplit(".", 1)
    return stem, f".{ext}"


def contains_core_weight(text: str) -> bool:
    """
    Check if text contains any core weight term.
    Uses substring matching to catch modified weights (SemiBold, ExtraLight, etc.)
    """
    if not text:
        return False

    text_upper = text.upper()
    for weight in CORE_WEIGHT_TERMS:
        if weight.upper() in text_upper:
            return True
    return False


def extract_width_prefix(style_part: str) -> Tuple[str, str]:
    """
    Extract width prefix if present at the start.
    Returns (width_prefix, remainder).
    """
    if not style_part:
        return "", ""

    # Sort by length to match longest first (e.g., "ExtraCondensed" before "Condensed")
    sorted_widths = sorted(WIDTH_TERMS, key=len, reverse=True)

    for width in sorted_widths:
        if style_part.startswith(width):
            return width, style_part[len(width) :]

    return "", style_part


def extract_slope_prefix(text: str) -> Tuple[str, str]:
    """
    Extract slope prefix if present at the start.
    Returns (slope_prefix, remainder).
    """
    if not text:
        return "", ""

    sorted_slopes = sorted(SLOPE_TERMS, key=len, reverse=True)

    for slope in sorted_slopes:
        if text.startswith(slope):
            return slope, text[len(slope) :]

    return "", text


# --- Regular Insertion Logic --------------------------------------------------------------------


def should_insert_regular(style_part: str) -> Tuple[bool, str]:
    """
    Determine if "Regular" should be inserted.
    Returns (should_insert, reason).

    Conservative approach: Only act on clear-cut cases.
    """
    if not style_part:
        # Empty style: Font.otf → Font-Regular.otf
        return True, "empty_style"

    # Check for weight terms - if present, definitely skip
    # This catches Bold, SemiBold, ExtraBold, etc. all with one check
    if contains_core_weight(style_part):
        return False, "has_weight"

    # Extract width if present
    width_prefix, after_width = extract_width_prefix(style_part)

    if width_prefix:
        # Case 1: Width only (e.g., "Condensed")
        if not after_width:
            return True, "width_only"

        # Case 2: Width + Slope (e.g., "CondensedItalic")
        slope_prefix, after_slope = extract_slope_prefix(after_width)
        if slope_prefix and not after_slope:
            return True, "width_slope"

        # Width + something unknown - skip to be safe
        return False, "width_unknown"

    # Case 3: Slope only (e.g., "Italic")
    slope_prefix, after_slope = extract_slope_prefix(style_part)
    if slope_prefix and not after_slope:
        return True, "slope_only"

    # Unknown pattern - skip and let human decide
    return False, "unknown_pattern"


def insert_regular(style_part: str) -> str:
    """
    Insert "Regular" in the appropriate position.
    Assumes should_insert_regular() returned True.
    """
    if not style_part:
        return "Regular"

    # Extract width if present
    width_prefix, after_width = extract_width_prefix(style_part)

    if width_prefix:
        # Check for slope after width
        slope_prefix, after_slope = extract_slope_prefix(after_width)
        if slope_prefix:
            # Width + Slope: Insert between them
            return f"{width_prefix}Regular{slope_prefix}{after_slope}"
        else:
            # Width only: Append Regular
            return f"{width_prefix}Regular"

    # Check for slope only
    slope_prefix, after_slope = extract_slope_prefix(style_part)
    if slope_prefix:
        # Slope only: Insert before it
        return f"Regular{slope_prefix}{after_slope}"

    # Shouldn't reach here if should_insert_regular() logic is correct
    return style_part


def build_new_filename(original_name: str) -> Tuple[str, bool, str]:
    """
    Process a filename and insert Regular if needed.
    Returns (new_filename, was_modified, reason).
    """
    stem, suffixes = split_stem_and_suffixes(original_name)
    if not stem:
        return original_name, False, "no_stem"

    # Split by last hyphen
    if "-" in stem:
        parts = stem.rsplit("-", 1)
        if parts[0] and parts[1]:
            family_part, style_part = parts[0], parts[1]
        else:
            # Malformed hyphen usage
            family_part, style_part = stem, ""
    else:
        # No hyphen: Check if stem contains weight term embedded in family name
        # e.g., "EdsMarketBoldScript" - don't add Regular to family names!
        if contains_core_weight(stem):
            return original_name, False, "weight_in_family_name"
        # No hyphen and no weight: entire stem is family, style is empty
        family_part, style_part = stem, ""

    should_insert, reason = should_insert_regular(style_part)

    if not should_insert:
        return original_name, False, reason

    new_style_part = insert_regular(style_part)

    # Reconstruct filename
    if new_style_part:
        final_stem = f"{family_part}-{new_style_part}"
    else:
        # Edge case: shouldn't happen, but handle gracefully
        final_stem = family_part

    final_name = f"{final_stem}{suffixes}"

    return final_name, True, reason


# --- Main Processing Logic ----------------------------------------------------------------------


@dataclass
class RenameDecision:
    old_name: str
    new_name: str
    was_modified: bool
    reason: str
    destination: Path


def compute_rename(file_path: Path) -> RenameDecision:
    old_name = file_path.name
    new_name, was_modified, reason = build_new_filename(old_name)
    return RenameDecision(
        old_name=old_name,
        new_name=new_name,
        was_modified=was_modified,
        reason=reason,
        destination=file_path.with_name(new_name),
    )


def iter_target_files(root: Path, recursive: bool) -> Iterable[Path]:
    """Iterate over target files, skipping hidden files and directories."""

    def _is_hidden(p: Path) -> bool:
        try:
            parts = p.relative_to(root).parts
        except ValueError:
            parts = p.parts
        return any(seg.startswith(".") for seg in parts)

    if root.is_file() and not _is_hidden(root):
        yield root
    elif root.is_dir():
        for p in root.rglob("*") if recursive else root.iterdir():
            if p.is_file() and not _is_hidden(p):
                yield p


def perform_rename(
    file_path: Path, *, dry_run: bool, conflict: str, verbose: bool, show_reason: bool
) -> Tuple[bool, str | None]:
    """Perform the actual file rename operation."""
    decision = compute_rename(file_path)

    if decision.new_name == decision.old_name:
        if verbose:
            msg = cs.StatusIndicator("unchanged").add_file(str(file_path))
            if show_reason:
                msg = msg.with_explanation(f"reason: {decision.reason}")
            msg.emit()
        return False, None

    destination = file_path.with_name(decision.new_name)
    if destination.exists():
        if conflict == "skip":
            if verbose:
                cs.StatusIndicator("warning").add_file(
                    decision.new_name
                ).with_explanation("exists, skipping").emit()
            return False, None
        if conflict == "unique":
            stem, suffixes = split_stem_and_suffixes(destination.name)
            counter = 1
            while destination.exists():
                destination = destination.with_name(f"{stem} ({counter}){suffixes}")
                counter += 1

    # Use same StatusIndicator for both dry-run and normal mode
    # DRY prefix will be added automatically when dry_run=True
    msg = (
        cs.StatusIndicator("updated", dry_run=dry_run)
        .add_message("rename")
        .add_values(old_value=decision.old_name, new_value=destination.name)
    )
    if show_reason:
        msg = msg.with_explanation(f"reason: {decision.reason}")
    msg.emit()

    if dry_run:
        return True, None

    try:
        file_path.rename(destination)
        return True, None
    except Exception as exc:
        return False, f"rename failed for {file_path}: {exc}"


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Insert 'Regular' into font filenames missing weight designation (minimalist approach).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
MINIMALIST APPROACH:
  This script uses core term matching with automatic modifier support:
  - Core weights: Thin, Light, Bold, Black, etc.
  - Modifiers automatically handled: Semi/Demi/Extra/Ultra + core weight
  - Example: "SemiBold" matches because it contains "Bold"
  
  Always run with --dry-run first and audit results before actual renaming.

Examples that WILL be changed:
  Font.otf → Font-Regular.otf
  Font-Italic.otf → Font-RegularItalic.otf
  Font-Condensed.otf → Font-CondensedRegular.otf
  Font-CondensedItalic.otf → Font-CondensedRegularItalic.otf

Examples that will be SKIPPED (require manual review):
  Font-Display.otf → unchanged (ambiguous term)
  Font-Text.otf → unchanged (ambiguous term)
  Font-Poster.otf → unchanged (ambiguous term)

Recommended workflow:
  1. python script.py /fonts --dry-run --show-reason > review.txt
  2. Review the output for false positives
  3. python script.py /fonts  (if satisfied)
  4. Manually handle any skipped ambiguous cases
        """,
    )
    parser.add_argument(
        "paths", nargs="+", help="File(s) and/or director(y/ies) to process"
    )
    parser.add_argument(
        "-r", "--recursive", action="store_true", help="Recurse into directories"
    )
    parser.add_argument(
        "-n", "--dry-run", action="store_true", help="Show changes without renaming"
    )
    parser.add_argument(
        "--conflict",
        choices=("skip", "unique", "overwrite"),
        default="unique",
        help="Action on name conflict (default: unique)",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Show unchanged files"
    )
    parser.add_argument(
        "--show-reason",
        action="store_true",
        help="Show reason for each decision (useful for auditing)",
    )
    parser.add_argument(
        "--add-weight",
        action="append",
        default=[],
        metavar="TERM",
        help="Add custom weight term to skip list",
    )
    parser.add_argument(
        "--add-slope",
        action="append",
        default=[],
        metavar="TERM",
        help="Add custom slope term",
    )
    parser.add_argument(
        "--add-width",
        action="append",
        default=[],
        metavar="TERM",
        help="Add custom width term",
    )
    return parser.parse_args(argv)


def load_user_terms(args: argparse.Namespace) -> None:
    """Add user-defined terms to dictionaries."""
    for term in args.add_weight or []:
        if term.strip():
            CORE_WEIGHT_TERMS.add(term.strip())

    for term in args.add_slope or []:
        if term.strip():
            SLOPE_TERMS.add(term.strip())

    for term in args.add_width or []:
        if term.strip():
            WIDTH_TERMS.add(term.strip())


def main(argv: Sequence[str]) -> int:
    """Main entry point."""
    args = parse_args(argv)
    load_user_terms(args)

    total_files, changed, errors = 0, 0, 0
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
                show_reason=args.show_reason,
            )
            if did_change:
                changed += 1
            if error_message is not None:
                errors += 1
                cs.StatusIndicator("error").with_explanation(error_message).emit()

    cs.fmt_processing_summary(
        dry_run=args.dry_run,
        updated=changed,
        unchanged=total_files - changed,
        errors=errors,
    )
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
