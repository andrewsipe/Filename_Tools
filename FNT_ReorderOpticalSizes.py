#!/usr/bin/env python3
"""
Reorder optical size terms in font filenames to appear before the hyphen.

Focus: Move optical size terms from the style part to the end of the family part.

Logic:
1. Split filename stem by the last hyphen into a "family" and "style" part.
2. Find all optical size terms in the style part and extract them in order found.
3. Remove the optical size terms from their original positions.
4. Rebuild as: FamilyName + OpticalSizeTerms - RemainingStyle
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Set, Tuple

import FontCore.core_console_styles as cs


# --- Optical Size Terms Dictionary -------------------------------------------------------------

# Common optical size terms
OPTICAL_SIZE_TERMS: Set[str] = {
    "Banner",
    "Big",
    "Caption",
    "Display",
    "Deck",
    "Fine",
    "Headline",
    "Head",
    "Large",
    "Micro",
    "Poster",
    "Subhead",
    "Small",
    "Titling",
    "Title",
    "Tall",
    "Text",
}


# --- Helpers ------------------------------------------------------------------------------------


def split_stem_and_suffixes(filename: str) -> Tuple[str, str]:
    """Split filename into stem and file extensions."""
    path = Path(filename)
    suffixes = "".join(path.suffixes)
    if not suffixes:
        return filename, ""
    return path.name[: -len(suffixes)], suffixes


# --- Optical Size Reordering Logic --------------------------------------------------------------


def reorder_optical_size_terms(
    family_part: str, style_part: str
) -> Tuple[str, str, List[str]]:
    """
    Extract optical size terms from style part and move them to the end of family part.
    Uses longest-match-first to handle compound terms.
    Returns (new_family_part, new_style_part, list_of_moved_optical_size_terms).
    """
    if not style_part:
        return family_part, style_part, []

    # Sort optical size terms by length (descending) to prioritize longer matches
    sorted_optical_terms = sorted(OPTICAL_SIZE_TERMS, key=len, reverse=True)

    # Track which parts of the string have been claimed by optical size terms
    claimed_ranges = []
    optical_matches = []  # (start, end, term) tuples

    # Find all optical size term matches, prioritizing longer ones
    for optical_term in sorted_optical_terms:
        start = 0
        while (pos := style_part.find(optical_term, start)) != -1:
            end = pos + len(optical_term)

            # Check if this range overlaps with any already claimed range
            overlaps = any(
                not (end <= claimed_start or pos >= claimed_end)
                for claimed_start, claimed_end in claimed_ranges
            )

            if not overlaps:
                optical_matches.append((pos, end, optical_term))
                claimed_ranges.append((pos, end))

            start = pos + 1

    # If no optical size terms found, no change needed
    if not optical_matches:
        return family_part, style_part, []

    # Sort matches by position in string to maintain order
    optical_matches.sort(key=lambda x: x[0])

    # Extract optical size terms in order they appear
    optical_terms_found = [term for _, _, term in optical_matches]

    # Build the remaining text by skipping the claimed ranges
    remaining_parts = []
    last_pos = 0

    for start, end, _ in optical_matches:
        # Add text before this optical size term
        if start > last_pos:
            remaining_parts.append(style_part[last_pos:start])
        last_pos = end

    # Add any remaining text after the last optical size term
    if last_pos < len(style_part):
        remaining_parts.append(style_part[last_pos:])

    remaining_text = "".join(remaining_parts)

    # Build new parts: optical size terms appended to family, remaining text as style
    new_family_part = family_part + "".join(optical_terms_found)
    new_style_part = remaining_text

    return new_family_part, new_style_part, optical_terms_found


def build_new_filename(original_name: str) -> Tuple[str, List[str]]:
    """Process a filename and reorder optical size terms."""
    stem, suffixes = split_stem_and_suffixes(original_name)
    if not stem:
        return original_name, []

    family_part, style_part = stem, ""
    if "-" in stem:
        parts = stem.rsplit("-", 1)
        if parts[0] and parts[1]:  # Ensure both parts are non-empty
            family_part, style_part = parts[0], parts[1]

    new_family_part, new_style_part, moved_optical = reorder_optical_size_terms(
        family_part, style_part
    )

    # Check if anything actually changed
    if new_family_part == family_part and new_style_part == style_part:
        return original_name, []

    # Rebuild filename
    if new_style_part:
        final_stem = f"{new_family_part}-{new_style_part}"
    else:
        final_stem = new_family_part

    final_name = f"{final_stem}{suffixes}"

    return final_name, moved_optical


# --- Main Processing Logic ----------------------------------------------------------------------


@dataclass
class RenameDecision:
    old_name: str
    new_name: str
    moved_optical: List[str]
    destination: Path


def compute_rename(file_path: Path) -> RenameDecision:
    old_name, (new_name, moved_optical) = (
        file_path.name,
        build_new_filename(file_path.name),
    )
    return RenameDecision(
        old_name=old_name,
        new_name=new_name,
        moved_optical=moved_optical,
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
        optical_info = (
            f" (moved: {', '.join(decision.moved_optical)})"
            if decision.moved_optical
            else ""
        )
        cs.StatusIndicator("info", dry_run=True).add_message(
            f"DRY-RUN rename{optical_info}"
        ).add_values(old_value=decision.old_name, new_value=destination.name).emit()
        return True, None

    try:
        file_path.rename(destination)
        optical_info = (
            f" (moved: {', '.join(decision.moved_optical)})"
            if decision.moved_optical
            else ""
        )
        cs.StatusIndicator("updated").add_message(f"rename{optical_info}").add_values(
            old_value=decision.old_name, new_value=destination.name
        ).emit()
        return True, None
    except Exception as exc:
        return False, f"rename failed for {file_path}: {exc}"


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Reorder optical size terms in font filenames to appear before the hyphen."
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
        "--add-optical",
        action="append",
        default=[],
        metavar="TERM",
        help="Add a custom optical size term to the dictionary",
    )
    return parser.parse_args(argv)


def load_user_optical_terms(args: argparse.Namespace) -> None:
    """Add user-defined optical size terms to the classification set."""
    for term in args.add_optical or []:
        if term.strip():
            OPTICAL_SIZE_TERMS.add(term.strip())


def main(argv: Sequence[str]) -> int:
    """Main entry point."""
    args = parse_args(argv)
    load_user_optical_terms(args)

    # Convert paths to Path objects
    path_objects = []
    for raw_path in args.paths:
        path = Path(raw_path)
        if path.exists():
            path_objects.append(path)
        else:
            cs.StatusIndicator("warning").add_file(raw_path).with_explanation(
                "path not found"
            ).emit()

    if not path_objects:
        cs.StatusIndicator("error").with_explanation("No valid paths to process").emit()
        return 1

    total_files, changed, errors = 0, 0, 0
    for path in path_objects:
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
