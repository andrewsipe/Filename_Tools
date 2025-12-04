#!/usr/bin/env python3
"""
Insert "Regular" after width terms when no weight is specified.

Focus: Add explicit "Regular" weight to width-only style specifications.

Logic:
1. Split filename stem by the last hyphen into a "family" and "style" part.
2. Check if style starts with a width term.
3. If width term is followed by nothing, a file extension, or a slope term, insert "Regular".
4. Examples:
   - Helvetica-Condensed.ttf → Helvetica-CondensedRegular.ttf
   - Helvetica-CondensedItalic.ttf → Helvetica-CondensedRegularItalic.ttf
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence, Set, Tuple

import core.core_console_styles as cs


# --- Width Terms Dictionary ---------------------------------------------------------------------

# Base width terms (the core 8)
BASE_WIDTH_TERMS: Set[str] = {
    "Compressed",
    "Compact",
    "Condensed",
    "Narrow",
    "Tight",
    "Wide",
    "Extended",
    "Expanded",
}

# Modifiers that combine with base terms
WIDTH_MODIFIERS: Set[str] = {
    "Semi",
    "Extra",
    "Ultra",
    "Super",
}


def generate_width_terms() -> Set[str]:
    """
    Generate all possible width term combinations.
    Returns base terms + all modifier+base combinations + X variations.
    """
    terms = set(BASE_WIDTH_TERMS)  # Start with base terms

    # Add all modifier + base combinations
    for modifier in WIDTH_MODIFIERS:
        for base in BASE_WIDTH_TERMS:
            terms.add(f"{modifier}{base}")

    # Add X variations (X, XX, XXX, etc. up to 7 X's)
    for base in BASE_WIDTH_TERMS:
        for x_count in range(1, 8):  # 1 to 7 X's
            terms.add(f"{'X' * x_count}{base}")

    return terms


WIDTH_TERMS: Set[str] = generate_width_terms()

# Slope/style terms that can follow width
SLOPE_TERMS: Set[str] = {
    "Italic",
    "Oblique",
    "Slanted",
    "Slant",
    "Inclined",
    "Backslanted",
    "Backslant",
    "Reverse",
    "Retalic",
    "Smallcaps",
}


# --- Helpers ------------------------------------------------------------------------------------


def split_stem_and_suffixes(filename: str) -> Tuple[str, str]:
    """Split filename into stem and file extensions."""
    path = Path(filename)
    suffixes = "".join(path.suffixes)
    if not suffixes:
        return filename, ""
    return path.name[: -len(suffixes)], suffixes


# --- Regular Insertion Logic -------------------------------------------------------------------


def insert_regular_after_width(style_part: str) -> Tuple[str, bool]:
    """
    Insert "Regular" after a width term if no weight is present.
    Returns (new_style_part, was_modified).
    """
    if not style_part:
        return "", False

    # Sort width terms by length (descending) to match longest first
    sorted_width_terms = sorted(WIDTH_TERMS, key=len, reverse=True)

    # Check if style starts with a width term
    matched_width = None
    for width_term in sorted_width_terms:
        if style_part.startswith(width_term):
            matched_width = width_term
            break

    # No width term at the start, no change needed
    if not matched_width:
        return style_part, False

    # Get the part after the width term
    after_width = style_part[len(matched_width) :]

    # Case 1: Width term is the entire style (e.g., "Condensed")
    if not after_width:
        return f"{matched_width}Regular", True

    # Case 2: Width term followed by a slope term (e.g., "CondensedItalic")
    # Check if what follows starts with a slope term
    matched_slope = None
    for slope_term in SLOPE_TERMS:
        if after_width.startswith(slope_term):
            matched_slope = slope_term
            break

    if matched_slope:
        # Insert Regular between width and slope
        remaining = after_width[len(matched_slope) :]
        return f"{matched_width}Regular{matched_slope}{remaining}", True

    # Case 3: Width term followed by something else (likely a weight)
    # Don't insert Regular - assume it's something like "CondensedBold"
    return style_part, False


def build_new_filename(original_name: str) -> Tuple[str, bool]:
    """Process a filename and insert Regular after width if needed."""
    stem, suffixes = split_stem_and_suffixes(original_name)
    if not stem:
        return original_name, False

    family_part, style_part = "", stem
    if "-" in stem:
        parts = stem.rsplit("-", 1)
        if parts[0] and parts[1]:  # Ensure both parts are non-empty
            family_part, style_part = parts[0], parts[1]

    new_style_part, was_modified = insert_regular_after_width(style_part)

    if not was_modified:
        return original_name, False

    final_stem = f"{family_part}-{new_style_part}" if family_part else new_style_part
    final_name = f"{final_stem}{suffixes}"

    return final_name, True


# --- Main Processing Logic ----------------------------------------------------------------------


@dataclass
class RenameDecision:
    old_name: str
    new_name: str
    was_modified: bool
    destination: Path


def compute_rename(file_path: Path) -> RenameDecision:
    old_name, (new_name, was_modified) = (
        file_path.name,
        build_new_filename(file_path.name),
    )
    return RenameDecision(
        old_name=old_name,
        new_name=new_name,
        was_modified=was_modified,
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
    file_path: Path, *, dry_run: bool, conflict: str, verbose: bool
) -> Tuple[bool, str | None]:
    """Perform the actual file rename operation."""
    decision = compute_rename(file_path)
    if decision.new_name == decision.old_name:
        if verbose:
            cs.StatusIndicator("unchanged").add_file(str(file_path)).emit()
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

    if dry_run:
        cs.StatusIndicator("info", dry_run=True).add_message(
            "DRY-RUN rename"
        ).add_values(old_value=decision.old_name, new_value=destination.name).emit()
        return True, None

    try:
        file_path.rename(destination)
        cs.StatusIndicator("updated").add_message("rename").add_values(
            old_value=decision.old_name, new_value=destination.name
        ).emit()
        return True, None
    except Exception as exc:
        return False, f"rename failed for {file_path}: {exc}"


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Insert 'Regular' after width terms when no weight is specified."
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
        help="On name conflict action (default: unique)",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Show unchanged files"
    )
    parser.add_argument(
        "--add-width",
        action="append",
        default=[],
        metavar="TERM",
        help="Add a custom width term to the dictionary",
    )
    parser.add_argument(
        "--add-slope",
        action="append",
        default=[],
        metavar="TERM",
        help="Add a custom slope/style term to the dictionary",
    )
    return parser.parse_args(argv)


def load_user_terms(args: argparse.Namespace) -> None:
    """Add user-defined width and slope terms."""
    for term in args.add_width or []:
        if term.strip():
            WIDTH_TERMS.add(term.strip())

    for term in args.add_slope or []:
        if term.strip():
            SLOPE_TERMS.add(term.strip())


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
