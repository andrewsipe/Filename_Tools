#!/usr/bin/env python3
"""
Reorder width terms in font filenames to appear immediately after the hyphen.

Focus: Move width terms to the beginning of the style part.

Logic:
1. Split filename stem by the last hyphen into a "family" and "style" part.
2. Find all width terms in the style part and extract them in order found.
3. Remove the width terms from their original positions.
4. Rebuild as: FamilyName-WidthTerms + RemainingStyle
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Set, Tuple

# Add project root to path for FontCore imports (works for root and subdirectory scripts)
_project_root = Path(__file__).parent
while not (_project_root / "FontCore").exists() and _project_root.parent != _project_root:
    _project_root = _project_root.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import FontCore.core_console_styles as cs
import FontCore.core_file_collector as collector


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
    "Demi",
    "Extra",
    "Ultra",
    "Super",
}


def generate_width_terms() -> Set[str]:
    """
    Generate all possible width term combinations.
    Returns base terms + all modifier+base combinations.
    Note: X-prefixed terms are handled dynamically via regex.
    """
    terms = set(BASE_WIDTH_TERMS)  # Start with base terms

    # Add all modifier + base combinations
    for modifier in WIDTH_MODIFIERS:
        for base in BASE_WIDTH_TERMS:
            terms.add(f"{modifier}{base}")

    return terms


WIDTH_TERMS: Set[str] = generate_width_terms()


# Build regex pattern for X-prefixed width terms (matches X, XX, XXX, etc.)
X_WIDTH_PATTERN = re.compile(
    r"(X+)(" + "|".join(re.escape(term) for term in BASE_WIDTH_TERMS) + r")"
)


# --- Helpers ------------------------------------------------------------------------------------


def split_stem_and_suffixes(filename: str) -> Tuple[str, str]:
    """Split filename into stem and file extensions."""
    path = Path(filename)
    # Use path.suffix to get only the actual file extension (last dot-separated part)
    # This avoids treating dots within the filename (e.g., "v0.1") as suffix separators
    suffix = path.suffix
    if not suffix:
        return filename, ""
    return path.stem, suffix


def is_variable_font(filename: str) -> bool:
    """Check if filename indicates a variable font (case-insensitive)."""
    stem, _ = split_stem_and_suffixes(filename)
    stem_lower = stem.lower()
    return any(indicator in stem_lower for indicator in ["variable", "var", "vf"])


# --- Width Reordering Logic --------------------------------------------------------------------


def extract_width_terms(text: str) -> Tuple[str, List[str]]:
    """
    Extract width terms from text and return the text with widths removed.
    Returns (text_without_widths, list_of_width_terms_found).
    """
    if not text:
        return "", []

    # Track which parts of the string have been claimed by width terms
    claimed_ranges = []
    width_matches = []  # (start, end, term) tuples

    # First pass: Find all X-prefixed width terms using regex
    for match in X_WIDTH_PATTERN.finditer(text):
        start, end = match.span()
        term = match.group(0)  # Full match (e.g., "XXXCondensed")

        # Check if this range overlaps with any already claimed range
        overlaps = any(
            not (end <= claimed_start or start >= claimed_end)
            for claimed_start, claimed_end in claimed_ranges
        )

        if not overlaps:
            width_matches.append((start, end, term))
            claimed_ranges.append((start, end))

    # Second pass: Find regular width terms (longest first)
    sorted_width_terms = sorted(WIDTH_TERMS, key=len, reverse=True)

    for width_term in sorted_width_terms:
        start_pos = 0
        while (pos := text.find(width_term, start_pos)) != -1:
            end = pos + len(width_term)

            # Check if this range overlaps with any already claimed range
            overlaps = any(
                not (end <= claimed_start or pos >= claimed_end)
                for claimed_start, claimed_end in claimed_ranges
            )

            if not overlaps:
                width_matches.append((pos, end, width_term))
                claimed_ranges.append((pos, end))

            start_pos = pos + 1

    # If no width terms found, return unchanged
    if not width_matches:
        return text, []

    # Sort matches by position in string to maintain order
    width_matches.sort(key=lambda x: x[0])

    # Extract width terms in order they appear
    width_terms_found = [term for _, _, term in width_matches]

    # Build the remaining text by skipping the claimed ranges
    remaining_parts = []
    last_pos = 0

    for start, end, _ in width_matches:
        # Add text before this width term
        if start > last_pos:
            remaining_parts.append(text[last_pos:start])
        last_pos = end

    # Add any remaining text after the last width term
    if last_pos < len(text):
        remaining_parts.append(text[last_pos:])

    remaining_text = "".join(remaining_parts)

    return remaining_text, width_terms_found


def extract_move_ahead_terms(text: str, terms: List[str]) -> Tuple[str, List[str]]:
    """
    Extract move-ahead terms from text using case-sensitive whole-word matching.
    Terms are extracted in the order they appear in the 'terms' list.
    Returns (text_without_terms, list_of_terms_found_in_order).

    Whole-word matching means the term must be followed by:
    - An uppercase letter (e.g., "AltBold" matches "Alt")
    - A digit (e.g., "Alt1" matches "Alt")
    - End of string
    But NOT by a lowercase letter (e.g., "Alter" doesn't match "Alt")
    """
    if not text or not terms:
        return text, []

    claimed_ranges = []
    term_matches = []  # (start, end, term, order) tuples

    # Search for each term in the order provided
    for order, term in enumerate(terms):
        start_pos = 0
        while (pos := text.find(term, start_pos)) != -1:
            end = pos + len(term)

            # Check whole-word boundary: must be followed by uppercase, digit, or end of string
            if end < len(text):
                next_char = text[end]
                # If followed by lowercase letter, it's part of another word
                if next_char.islower():
                    start_pos = pos + 1
                    continue

            # Check if this range overlaps with any already claimed range
            overlaps = any(
                not (end <= claimed_start or pos >= claimed_end)
                for claimed_start, claimed_end in claimed_ranges
            )

            if not overlaps:
                term_matches.append((pos, end, term, order))
                claimed_ranges.append((pos, end))

            start_pos = pos + 1

    if not term_matches:
        return text, []

    # Sort by order in the terms list (respecting flag order), then by position in text
    term_matches.sort(key=lambda x: (x[3], x[0]))

    # Extract terms in the order they were specified in flags
    terms_found = [term for _, _, term, _ in term_matches]

    # Build the remaining text by skipping the claimed ranges
    # Sort by position for text reconstruction
    position_sorted = sorted(term_matches, key=lambda x: x[0])
    remaining_parts = []
    last_pos = 0

    for start, end, _, _ in position_sorted:
        if start > last_pos:
            remaining_parts.append(text[last_pos:start])
        last_pos = end

    if last_pos < len(text):
        remaining_parts.append(text[last_pos:])

    remaining_text = "".join(remaining_parts)

    return remaining_text, terms_found


def extract_number_terms(
    text: str, prefix: str | None, family_part: str, ignore_patterns: List[str]
) -> Tuple[str, List[str]]:
    """
    Extract number terms from text with optional prefix.
    Supports prefix+digits (e.g., "No87") and digits+prefix (e.g., "87No").
    If prefix is empty string, extracts all numbers (integers and decimals) from anywhere in text.
    Returns (text_without_numbers, list_of_number_terms_found).

    When prefix is provided, respects word boundaries: "87NoBold" matches, "87North" doesn't.
    When prefix is empty string, extracts all numbers regardless of surrounding characters.
    Checks family part against literal ignore patterns before extracting.
    """
    if not text:
        return text, []

    # Check if family part matches any ignore patterns (literal match)
    if family_part and ignore_patterns:
        for pattern in ignore_patterns:
            if pattern in family_part:
                return text, []

    claimed_ranges = []
    number_matches = []  # (start, end, term) tuples

    if prefix:
        # Build patterns for prefix+digits and digits+prefix
        # Pattern: prefix followed by digits (e.g., "No87")
        prefix_digits_pattern = re.compile(
            r"(" + re.escape(prefix) + r"\d+)(?=[A-Z]|\d|$)"
        )
        # Pattern: digits followed by prefix (e.g., "87No")
        digits_prefix_pattern = re.compile(
            r"(\d+" + re.escape(prefix) + r")(?=[A-Z]|\d|$)"
        )

        # Find prefix+digits matches
        for match in prefix_digits_pattern.finditer(text):
            start, end = match.span()
            term = match.group(1)

            # Check if followed by a lowercase letter (word boundary check)
            if end < len(text) and text[end].islower():
                continue

            overlaps = any(
                not (end <= claimed_start or start >= claimed_end)
                for claimed_start, claimed_end in claimed_ranges
            )

            if not overlaps:
                number_matches.append((start, end, term))
                claimed_ranges.append((start, end))

        # Find digits+prefix matches
        for match in digits_prefix_pattern.finditer(text):
            start, end = match.span()
            term = match.group(1)

            # Check if followed by a lowercase letter (word boundary check)
            if end < len(text) and text[end].islower():
                continue

            overlaps = any(
                not (end <= claimed_start or start >= claimed_end)
                for claimed_start, claimed_end in claimed_ranges
            )

            if not overlaps:
                number_matches.append((start, end, term))
                claimed_ranges.append((start, end))
    else:
        # No prefix - match all numbers (integers and decimals) from anywhere in text
        # Pattern matches: digits, optionally followed by decimal point and more digits
        standalone_pattern = re.compile(r"(\d+\.?\d*)")

        for match in standalone_pattern.finditer(text):
            start, end = match.span()
            term = match.group(1)

            overlaps = any(
                not (end <= claimed_start or start >= claimed_end)
                for claimed_start, claimed_end in claimed_ranges
            )

            if not overlaps:
                number_matches.append((start, end, term))
                claimed_ranges.append((start, end))

    if not number_matches:
        return text, []

    # Sort matches by position in string to maintain order
    number_matches.sort(key=lambda x: x[0])

    # Extract number terms in order they appear
    number_terms_found = [term for _, _, term in number_matches]

    # Build the remaining text by skipping the claimed ranges
    remaining_parts = []
    last_pos = 0

    for start, end, _ in number_matches:
        if start > last_pos:
            remaining_parts.append(text[last_pos:start])
        last_pos = end

    if last_pos < len(text):
        remaining_parts.append(text[last_pos:])

    remaining_text = "".join(remaining_parts)

    return remaining_text, number_terms_found


def build_new_filename(
    original_name: str,
    move_ahead_terms: List[str] | None = None,
    number_prefix: str | None = None,
    ignore_families: List[str] | None = None,
) -> Tuple[str, dict]:
    """
    Process a filename and reorder terms (numbers, move-ahead, widths).

    Returns (new_filename, dict_of_moved_terms) where dict contains:
    - 'numbers': List of moved number terms
    - 'move_ahead': List of moved ahead terms
    - 'widths': List of moved width terms
    """
    # Skip variable fonts
    if is_variable_font(original_name):
        return original_name, {"numbers": [], "move_ahead": [], "widths": []}

    stem, suffixes = split_stem_and_suffixes(original_name)
    if not stem:
        return original_name, {"numbers": [], "move_ahead": [], "widths": []}

    move_ahead_terms = move_ahead_terms or []
    ignore_families = ignore_families or []

    # If there's a hyphen, split on it
    if "-" in stem:
        parts = stem.rsplit("-", 1)
        if parts[0] and parts[1]:  # Ensure both parts are non-empty
            family_part, style_part = parts[0], parts[1]

            # Extract all term types from BOTH parts
            # Order: numbers -> move_ahead -> widths

            # Extract number terms (if configured)
            clean_family = family_part
            clean_style = style_part
            all_numbers = []

            if number_prefix is not None:  # Can be empty string for standalone numbers
                clean_family, family_numbers = extract_number_terms(
                    family_part, number_prefix, family_part, ignore_families
                )
                clean_style, style_numbers = extract_number_terms(
                    style_part, number_prefix, family_part, ignore_families
                )
                all_numbers = family_numbers + style_numbers

            # Extract move-ahead terms (if configured)
            all_move_ahead = []
            if move_ahead_terms:
                clean_family, family_move_ahead = extract_move_ahead_terms(
                    clean_family, move_ahead_terms
                )
                clean_style, style_move_ahead = extract_move_ahead_terms(
                    clean_style, move_ahead_terms
                )
                all_move_ahead = family_move_ahead + style_move_ahead

            # Extract width terms (always)
            clean_family, family_widths = extract_width_terms(clean_family)
            clean_style, style_widths = extract_width_terms(clean_style)
            all_widths = family_widths + style_widths

            # If no terms found at all, return unchanged
            if not all_numbers and not all_move_ahead and not all_widths:
                return original_name, {"numbers": [], "move_ahead": [], "widths": []}

            # Build final name: CleanFamily-Numbers+MoveAhead+Widths+CleanStyle
            # Order: Numbers first, then move-ahead, then widths
            all_terms = (
                "".join(all_numbers) + "".join(all_move_ahead) + "".join(all_widths)
            )
            final_stem = f"{clean_family}-{all_terms}{clean_style}"
            final_name = f"{final_stem}{suffixes}"

            return final_name, {
                "numbers": all_numbers,
                "move_ahead": all_move_ahead,
                "widths": all_widths,
            }

    # No hyphen - extract terms and insert hyphen before them
    clean_stem = stem
    all_numbers = []
    all_move_ahead = []

    if number_prefix is not None:
        clean_stem, all_numbers = extract_number_terms(
            clean_stem, number_prefix, stem, ignore_families
        )

    if move_ahead_terms:
        clean_stem, all_move_ahead = extract_move_ahead_terms(
            clean_stem, move_ahead_terms
        )

    clean_stem, all_widths = extract_width_terms(clean_stem)

    # If we found any terms, add hyphen and reorder
    if all_numbers or all_move_ahead or all_widths:
        all_terms = "".join(all_numbers) + "".join(all_move_ahead) + "".join(all_widths)
        final_stem = f"{clean_stem}-{all_terms}"
        final_name = f"{final_stem}{suffixes}"
        return final_name, {
            "numbers": all_numbers,
            "move_ahead": all_move_ahead,
            "widths": all_widths,
        }

    return original_name, {"numbers": [], "move_ahead": [], "widths": []}


# --- Analysis & Preview -------------------------------------------------------------------------


def analyze_files(
    paths: List[Path],
    recursive: bool,
    move_ahead_terms: List[str] | None = None,
    number_prefix: str | None = None,
    ignore_families: List[str] | None = None,
) -> dict:
    """
    Analyze files to determine what terms will be affected.
    Returns dict with analysis data.
    """
    number_terms_found = set()
    move_ahead_terms_found = set()
    width_terms_found = set()
    files_with_changes = []

    # Collect all font files using the file collector
    font_files = collector.collect_font_files(
        paths=[str(p) for p in paths], recursive=recursive
    )

    # Analyze each file
    for file_path_str in font_files:
        file_path = Path(file_path_str)
        decision = compute_rename(
            file_path, move_ahead_terms, number_prefix, ignore_families
        )
        # Only include files where the name actually changes
        if decision.all_moved_terms() and decision.new_name != decision.old_name:
            files_with_changes.append((file_path, decision))
            number_terms_found.update(decision.moved_numbers)
            move_ahead_terms_found.update(decision.moved_ahead)
            width_terms_found.update(decision.moved_widths)

    # Sort files_with_changes alphabetically by filename
    files_with_changes.sort(key=lambda x: x[0].name.lower())

    return {
        "total_files": len(font_files),
        "files_with_changes": files_with_changes,
        "number_terms_found": sorted(number_terms_found),
        "move_ahead_terms_found": sorted(move_ahead_terms_found),
        "width_terms_found": sorted(width_terms_found),
    }


def show_preflight_preview(analysis: dict) -> None:
    """Display a preview of what will be changed."""
    cs.emit("")
    cs.StatusIndicator("info").add_message("Term Reordering Preview").emit()

    # Show statistics
    cs.emit(
        f"{cs.indent(1)}Total files scanned: {cs.fmt_count(analysis['total_files'])}"
    )
    cs.emit(
        f"{cs.indent(1)}Files requiring changes: {cs.fmt_count(len(analysis['files_with_changes']))}"
    )

    # Show terms that will be addressed by type
    if analysis["number_terms_found"]:
        cs.emit(
            f"{cs.indent(1)}Number terms to reorder: {', '.join(analysis['number_terms_found'])}"
        )
    if analysis["move_ahead_terms_found"]:
        cs.emit(
            f"{cs.indent(1)}Move-ahead terms to reorder: {', '.join(analysis['move_ahead_terms_found'])}"
        )
    if analysis["width_terms_found"]:
        cs.emit(
            f"{cs.indent(1)}Width terms to reorder: {', '.join(analysis['width_terms_found'])}"
        )

    cs.emit("")

    # Show ALL changes with highlighted terms
    if analysis["files_with_changes"]:
        cs.StatusIndicator("info").add_message("All changes:").emit()

        if cs.RICH_AVAILABLE:
            table = cs.create_table(show_header=True)
            if table:
                table.add_column("Original", style="lighttext", no_wrap=False)
                table.add_column("New Name", style="lighttext", no_wrap=False)
                table.add_column("Moved", style="cyan", min_width=12)

                for file_path, decision in analysis["files_with_changes"]:
                    # Combine all moved terms for highlighting
                    all_moved = decision.all_moved_terms()

                    # Build highlighted before/after strings
                    before_highlighted = highlight_width_terms_in_filename(
                        decision.old_name, all_moved, style="before"
                    )
                    after_highlighted = highlight_width_terms_in_filename(
                        decision.new_name,
                        all_moved,
                        style="after",
                        mark_hyphen=True,
                    )

                    # Build moved terms info with type labels
                    moved_parts = []
                    if decision.moved_numbers:
                        moved_parts.append(f"#: {', '.join(decision.moved_numbers)}")
                    if decision.moved_ahead:
                        moved_parts.append(f"↑: {', '.join(decision.moved_ahead)}")
                    if decision.moved_widths:
                        moved_parts.append(f"Width: {', '.join(decision.moved_widths)}")
                    moved_info = "; ".join(moved_parts)

                    table.add_row(before_highlighted, after_highlighted, moved_info)

                console = cs.get_console()
                console.print(table)
        else:
            # Fallback for non-Rich environments
            for file_path, decision in analysis["files_with_changes"]:
                moved_parts = []
                if decision.moved_numbers:
                    moved_parts.append(f"numbers: {', '.join(decision.moved_numbers)}")
                if decision.moved_ahead:
                    moved_parts.append(f"move-ahead: {', '.join(decision.moved_ahead)}")
                if decision.moved_widths:
                    moved_parts.append(f"widths: {', '.join(decision.moved_widths)}")
                moved_info = f" (moved: {'; '.join(moved_parts)})"
                cs.emit(
                    f"{cs.indent(1)}{decision.old_name} -> {decision.new_name}{moved_info}"
                )

    cs.emit("")


def highlight_width_terms_in_filename(
    filename: str,
    width_terms: List[str],
    style: str = "before",
    mark_hyphen: bool = False,
) -> str:
    """
    Highlight width terms in a filename with the specified style.

    Args:
        filename: The filename to process
        width_terms: List of width terms to highlight
        style: "before" (turquoise) or "after" (magenta)
        mark_hyphen: If True, also highlight the hyphen after family name (for after style)
    """
    if not cs.RICH_AVAILABLE or not width_terms:
        return filename

    result = filename

    # For "after" style, highlight the hyphen first if needed
    if mark_hyphen and style == "after" and "-" in result:
        # Only highlight the first hyphen (the one after family name)
        result = result.replace("-", "[value.after]-[/value.after]", 1)

    # Highlight each width term (all occurrences)
    for term in width_terms:
        if style == "before":
            # Highlight all occurrences in the "before" filename
            result = result.replace(term, f"[value.before]{term}[/value.before]")
        else:  # after
            # Highlight all occurrences in the "after" filename
            result = result.replace(term, f"[value.after]{term}[/value.after]")

    return result


# --- Main Processing Logic ----------------------------------------------------------------------


@dataclass
class RenameDecision:
    old_name: str
    new_name: str
    moved_numbers: List[str]
    moved_ahead: List[str]
    moved_widths: List[str]
    destination: Path

    def all_moved_terms(self) -> List[str]:
        """Return all moved terms in order: numbers, move-ahead, widths."""
        return self.moved_numbers + self.moved_ahead + self.moved_widths


def compute_rename(
    file_path: Path,
    move_ahead_terms: List[str] | None = None,
    number_prefix: str | None = None,
    ignore_families: List[str] | None = None,
) -> RenameDecision:
    """
    Compute the rename decision for a file with optional term reordering.
    """
    old_name = file_path.name
    new_name, moved_terms = build_new_filename(
        old_name, move_ahead_terms, number_prefix, ignore_families
    )

    return RenameDecision(
        old_name=old_name,
        new_name=new_name,
        moved_numbers=moved_terms["numbers"],
        moved_ahead=moved_terms["move_ahead"],
        moved_widths=moved_terms["widths"],
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
    file_path: Path,
    *,
    dry_run: bool,
    conflict: str,
    verbose: bool,
    move_ahead_terms: List[str] | None = None,
    number_prefix: str | None = None,
    ignore_families: List[str] | None = None,
) -> Tuple[bool, str | None]:
    """Perform the actual file rename operation."""
    decision = compute_rename(
        file_path, move_ahead_terms, number_prefix, ignore_families
    )
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

    # Build info string for moved terms
    moved_parts = []
    if decision.moved_numbers:
        moved_parts.append(f"#: {', '.join(decision.moved_numbers)}")
    if decision.moved_ahead:
        moved_parts.append(f"↑: {', '.join(decision.moved_ahead)}")
    if decision.moved_widths:
        moved_parts.append(f"Width: {', '.join(decision.moved_widths)}")
    terms_info = f" ({'; '.join(moved_parts)})" if moved_parts else ""

    if dry_run:
        cs.StatusIndicator("info", dry_run=True).add_message(
            f"DRY-RUN rename{terms_info}"
        ).add_values(old_value=decision.old_name, new_value=destination.name).emit()
        return True, None

    try:
        file_path.rename(destination)
        cs.StatusIndicator("updated").add_message("rename" + terms_info).add_values(
            old_value=decision.old_name, new_value=destination.name
        ).emit()
        return True, None
    except Exception as exc:
        return False, f"rename failed for {file_path}: {exc}"


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Reorder width terms in font filenames to appear after the hyphen."
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
        "--no-preview",
        action="store_true",
        help="Skip preflight preview and proceed directly",
    )
    parser.add_argument(
        "--add-width",
        action="append",
        default=[],
        metavar="TERM",
        help="Add a custom width term to the dictionary",
    )
    parser.add_argument(
        "--move-ahead",
        action="append",
        default=[],
        metavar="TERM",
        help="Move specific term(s) ahead of width terms (repeatable, supports comma-separated)",
    )
    parser.add_argument(
        "--number-term",
        nargs="?",
        const="",
        metavar="TERM",
        help="Extract number patterns paired with this term (e.g., 'No' matches 'No87' or '87No', 'G' matches 'G1' or '1G'). If used without a value, reflows all numbers (including decimals) to the front of the filename.",
    )
    parser.add_argument(
        "--do-not-move",
        action="append",
        default=[],
        metavar="PATTERN",
        help="Don't extract numbers from family names matching these literal patterns (repeatable, supports comma-separated)",
    )
    return parser.parse_args(argv)


def load_user_width_terms(args: argparse.Namespace) -> None:
    """Add user-defined width terms to the classification set."""
    for term in args.add_width or []:
        if term.strip():
            WIDTH_TERMS.add(term.strip())


def parse_comma_separated_list(items: List[str]) -> List[str]:
    """
    Parse list that may contain comma-separated values.
    Preserves order and handles both --flag val1,val2 and --flag val1 --flag val2.
    """
    result = []
    for item in items:
        # Split on comma and strip whitespace
        parts = [part.strip() for part in item.split(",") if part.strip()]
        result.extend(parts)
    return result


def main(argv: Sequence[str]) -> int:
    """Main entry point."""
    args = parse_args(argv)
    load_user_width_terms(args)

    # Parse comma-separated configuration lists
    move_ahead_terms = parse_comma_separated_list(args.move_ahead)
    ignore_families = parse_comma_separated_list(args.do_not_move)

    # number_term can be None (not set), empty string (standalone numbers), or a term
    # If --number-term was used without a value but followed by a path, argparse may
    # have consumed the first path as the value. Detect and fix this.
    number_term = args.number_term
    if number_term and (
        number_term.startswith("/")
        or number_term.startswith("./")
        or ("/" in number_term and len(number_term) > 3)
    ):
        # This looks like a path, not a term - treat as empty string and add to paths
        args.paths.insert(0, number_term)
        number_term = ""

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

    # Analyze files and show preview unless --no-preview
    if not args.no_preview:
        analysis = analyze_files(
            path_objects,
            args.recursive,
            move_ahead_terms,
            number_term,
            ignore_families,
        )

        if analysis["files_with_changes"]:
            show_preflight_preview(analysis)

            # Ask for confirmation unless dry-run
            if not args.dry_run:
                if not cs.prompt_confirm(
                    "Ready to reorder terms in filenames",
                    action_prompt="Proceed with renaming?",
                    default=True,
                ):
                    cs.StatusIndicator("info").add_message("Operation cancelled").emit()
                    return 0
        else:
            cs.StatusIndicator("info").add_message(
                "No files require term reordering"
            ).emit()
            return 0

    # Process files
    total_files, changed, errors = 0, 0, 0
    for path in path_objects:
        for file_path in iter_target_files(path, args.recursive):
            total_files += 1
            did_change, error_message = perform_rename(
                file_path,
                dry_run=args.dry_run,
                conflict=args.conflict,
                verbose=args.verbose,
                move_ahead_terms=move_ahead_terms,
                number_prefix=number_term,
                ignore_families=ignore_families,
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
