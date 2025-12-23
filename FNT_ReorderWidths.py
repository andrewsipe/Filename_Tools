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
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Set, Tuple

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
import FontCore.core_file_collector as collector
import re


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


# --- TermExtractor Class ---------------------------------------------------------------------


class TermExtractor:
    """
    Extracts and manages terms in font filenames with support for:
    - Primary terms (widths, slopes, etc.) with X-prefix support
    - Move-ahead terms (positioned before primary terms)
    - Move-behind terms (positioned after primary terms)
    - Number terms with optional prefix (positioned before or after primary terms)
    """

    def __init__(
        self,
        base_terms: Set[str],
        modifiers: Set[str] | None = None,
        support_x_prefix: bool = True,
    ):
        """
        Initialize the term extractor.

        Args:
            base_terms: Core terms to extract (e.g., {"Condensed", "Italic"})
            modifiers: Optional modifiers that combine with base terms (e.g., {"Semi", "Extra"})
            support_x_prefix: Whether to support X-prefixed terms (e.g., "XXCondensed")
        """
        self.base_terms = base_terms
        self.modifiers = modifiers or set()
        self.support_x_prefix = support_x_prefix

        # Generate all term combinations
        self.all_terms = self._generate_term_combinations()

        # Build X-prefix pattern if needed
        self.x_pattern = self._build_x_prefix_pattern() if support_x_prefix else None

    def _generate_term_combinations(self) -> Set[str]:
        """Generate all possible term combinations (base + modifier+base)."""
        terms = set(self.base_terms)

        # Add all modifier + base combinations
        for modifier in self.modifiers:
            for base in self.base_terms:
                terms.add(f"{modifier}{base}")

        return terms

    def _build_x_prefix_pattern(self) -> re.Pattern | None:
        """Build regex pattern for X-prefixed terms (matches X, XX, XXX, etc.)."""
        if not self.base_terms:
            return None
        return re.compile(
            r"(X+)(" + "|".join(re.escape(term) for term in self.base_terms) + r")"
        )

    def extract_primary_terms(self, text: str) -> Tuple[str, List[str]]:
        """
        Extract primary terms (e.g., width or slope terms) from text.
        Returns (text_without_terms, list_of_terms_found).
        """
        if not text:
            return "", []

        claimed_ranges = []
        term_matches = []  # (start, end, term) tuples

        # First pass: Find X-prefixed terms if supported
        if self.x_pattern:
            for match in self.x_pattern.finditer(text):
                start, end = match.span()
                term = match.group(0)

                if not self._overlaps_claimed(start, end, claimed_ranges):
                    term_matches.append((start, end, term))
                    claimed_ranges.append((start, end))

        # Second pass: Find regular terms (longest first to avoid partial matches)
        sorted_terms = sorted(self.all_terms, key=len, reverse=True)

        for term in sorted_terms:
            start_pos = 0
            while (pos := text.find(term, start_pos)) != -1:
                end = pos + len(term)

                if not self._overlaps_claimed(pos, end, claimed_ranges):
                    term_matches.append((pos, end, term))
                    claimed_ranges.append((pos, end))

                start_pos = pos + 1

        if not term_matches:
            return text, []

        # Sort by position to maintain order
        term_matches.sort(key=lambda x: x[0])

        # Extract terms in order they appear
        terms_found = [term for _, _, term in term_matches]

        # Build remaining text
        remaining_text = self._build_remaining_text(text, term_matches)

        return remaining_text, terms_found

    @staticmethod
    def extract_auxiliary_terms(text: str, terms: List[str]) -> Tuple[str, List[str]]:
        """
        Extract auxiliary terms (move-ahead/move-behind) from text using case-sensitive whole-word matching.
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
                if not TermExtractor._overlaps_claimed(pos, end, claimed_ranges):
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
        remaining_text = TermExtractor._build_remaining_text(
            text, [(s, e, t) for s, e, t, _ in position_sorted]
        )

        return remaining_text, terms_found

    @staticmethod
    def extract_number_terms(
        text: str,
        prefix: str | None,
        family_part: str,
        ignore_patterns: List[str],
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

                if not TermExtractor._overlaps_claimed(start, end, claimed_ranges):
                    number_matches.append((start, end, term))
                    claimed_ranges.append((start, end))

            # Find digits+prefix matches
            for match in digits_prefix_pattern.finditer(text):
                start, end = match.span()
                term = match.group(1)

                # Check if followed by a lowercase letter (word boundary check)
                if end < len(text) and text[end].islower():
                    continue

                if not TermExtractor._overlaps_claimed(start, end, claimed_ranges):
                    number_matches.append((start, end, term))
                    claimed_ranges.append((start, end))
        else:
            # No prefix - match all numbers (integers and decimals) from anywhere in text
            # Pattern matches: digits, optionally followed by decimal point and more digits
            standalone_pattern = re.compile(r"(\d+\.?\d*)")

            for match in standalone_pattern.finditer(text):
                start, end = match.span()
                term = match.group(1)

                if not TermExtractor._overlaps_claimed(start, end, claimed_ranges):
                    number_matches.append((start, end, term))
                    claimed_ranges.append((start, end))

        if not number_matches:
            return text, []

        # Sort matches by position in string to maintain order
        number_matches.sort(key=lambda x: x[0])

        # Extract number terms in order they appear
        number_terms_found = [term for _, _, term in number_matches]

        # Build the remaining text by skipping the claimed ranges
        remaining_text = TermExtractor._build_remaining_text(text, number_matches)

        return remaining_text, number_terms_found

    @staticmethod
    def format_number_term_with_leading_zero(term: str, leading_zero: bool) -> str:
        """
        Format a number term with leading zero if requested and applicable.

        Handles:
        - Standalone numbers: "1" -> "01", "10" -> "10", "1.5" -> "1.5"
        - Prefix+digits: "No1" -> "No01", "No87" -> "No87"
        - Digits+prefix: "1No" -> "01No", "87No" -> "87No"

        Only formats single-digit integers (1-9), leaves multi-digit and decimals unchanged.
        """
        if not leading_zero:
            return term

        # Check if it's a standalone number (just digits, possibly with decimal)
        if re.match(r"^\d+\.?\d*$", term):
            try:
                # Try to parse as integer first
                num = int(term)
                if 1 <= num <= 9:
                    return f"{num:02d}"  # 1 -> "01", 9 -> "09"
            except ValueError:
                # Has decimal point, leave unchanged
                pass
            return term

        # Check for prefix+digits pattern (e.g., "No87", "G1")
        prefix_digits_match = re.match(r"^([A-Za-z]+)(\d+)$", term)
        if prefix_digits_match:
            prefix = prefix_digits_match.group(1)
            digits = prefix_digits_match.group(2)
            try:
                num = int(digits)
                if 1 <= num <= 9:
                    return f"{prefix}{num:02d}"  # "No1" -> "No01"
            except ValueError:
                pass
            return term

        # Check for digits+prefix pattern (e.g., "87No", "1G")
        digits_prefix_match = re.match(r"^(\d+)([A-Za-z]+)$", term)
        if digits_prefix_match:
            digits = digits_prefix_match.group(1)
            prefix = digits_prefix_match.group(2)
            try:
                num = int(digits)
                if 1 <= num <= 9:
                    return f"{num:02d}{prefix}"  # "1No" -> "01No"
            except ValueError:
                pass
            return term

        # Doesn't match expected patterns, return unchanged
        return term

    @staticmethod
    def _overlaps_claimed(
        start: int, end: int, claimed_ranges: List[Tuple[int, int]]
    ) -> bool:
        """Check if a range overlaps with any claimed ranges."""
        return any(
            not (end <= claimed_start or start >= claimed_end)
            for claimed_start, claimed_end in claimed_ranges
        )

    @staticmethod
    def _build_remaining_text(
        text: str, term_matches: List[Tuple[int, int, str]]
    ) -> str:
        """Build text with matched terms removed."""
        remaining_parts = []
        last_pos = 0

        for start, end, _ in term_matches:
            # Add text before this term
            if start > last_pos:
                remaining_parts.append(text[last_pos:start])
            last_pos = end

        # Add any remaining text after the last term
        if last_pos < len(text):
            remaining_parts.append(text[last_pos:])

        return "".join(remaining_parts)


# --- Create Width Term Extractor Instance ---------------------------------------------------------------------

# Create width term extractor instance
WIDTH_EXTRACTOR = TermExtractor(
    base_terms=BASE_WIDTH_TERMS,
    modifiers=WIDTH_MODIFIERS,
    support_x_prefix=True,
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
    """Extract width terms from text and return the text with widths removed."""
    return WIDTH_EXTRACTOR.extract_primary_terms(text)


def create_empty_moved_terms_dict() -> dict:
    """Create empty dictionary for moved terms tracking."""
    return {
        "numbers_ahead": [],
        "numbers_behind": [],
        "move_ahead": [],
        "widths": [],
        "move_behind": [],
    }


def apply_leading_zero_to_list(terms: List[str], leading_zero: bool) -> List[str]:
    """Apply leading zero formatting to a list of number terms if requested."""
    if not leading_zero:
        return terms
    return [
        TermExtractor.format_number_term_with_leading_zero(term, True) for term in terms
    ]


def extract_all_terms_from_parts(
    family_part: str,
    style_part: str,
    move_ahead_terms: List[str],
    move_behind_terms: List[str],
    number_ahead_prefix: str | None,
    number_behind_prefix: str | None,
    ignore_families: List[str],
    leading_zero: bool,
) -> Tuple[str, str, dict]:
    """
    Extract all configured term types from family and style parts.
    Returns (clean_family, clean_style, moved_terms_dict).
    """
    clean_family = family_part
    clean_style = style_part
    moved_terms = create_empty_moved_terms_dict()

    # Extract in order: numbers_ahead → move_ahead → widths → move_behind → numbers_behind

    # Extract numbers_ahead
    if number_ahead_prefix is not None:
        clean_family, family_nums = TermExtractor.extract_number_terms(
            clean_family, number_ahead_prefix, family_part, ignore_families
        )
        clean_style, style_nums = TermExtractor.extract_number_terms(
            clean_style, number_ahead_prefix, family_part, ignore_families
        )
        moved_terms["numbers_ahead"] = apply_leading_zero_to_list(
            family_nums + style_nums, leading_zero
        )

    # Extract move_ahead terms
    if move_ahead_terms:
        clean_family, family_move_ahead = TermExtractor.extract_auxiliary_terms(
            clean_family, move_ahead_terms
        )
        clean_style, style_move_ahead = TermExtractor.extract_auxiliary_terms(
            clean_style, move_ahead_terms
        )
        moved_terms["move_ahead"] = family_move_ahead + style_move_ahead

    # Extract width terms (always)
    clean_family, family_widths = extract_width_terms(clean_family)
    clean_style, style_widths = extract_width_terms(clean_style)
    moved_terms["widths"] = family_widths + style_widths

    # Extract move_behind terms
    if move_behind_terms:
        clean_family, family_move_behind = TermExtractor.extract_auxiliary_terms(
            clean_family, move_behind_terms
        )
        clean_style, style_move_behind = TermExtractor.extract_auxiliary_terms(
            clean_style, move_behind_terms
        )
        moved_terms["move_behind"] = family_move_behind + style_move_behind

    # Extract numbers_behind
    if number_behind_prefix is not None:
        clean_family, family_nums = TermExtractor.extract_number_terms(
            clean_family, number_behind_prefix, family_part, ignore_families
        )
        clean_style, style_nums = TermExtractor.extract_number_terms(
            clean_style, number_behind_prefix, family_part, ignore_families
        )
        moved_terms["numbers_behind"] = apply_leading_zero_to_list(
            family_nums + style_nums, leading_zero
        )

    return clean_family, clean_style, moved_terms


def build_new_filename(
    original_name: str,
    move_ahead_terms: List[str] | None = None,
    move_behind_terms: List[str] | None = None,
    number_ahead_prefix: str | None = None,
    number_behind_prefix: str | None = None,
    ignore_families: List[str] | None = None,
    leading_zero: bool = False,
) -> Tuple[str, dict]:
    """
    Process a filename and reorder terms (numbers, move-ahead, widths, move-behind).

    Returns (new_filename, dict_of_moved_terms) where dict contains:
    - 'numbers_ahead': List of moved number terms (before width)
    - 'numbers_behind': List of moved number terms (after width)
    - 'move_ahead': List of moved ahead terms
    - 'widths': List of moved width terms
    - 'move_behind': List of moved behind terms
    """
    # Skip variable fonts
    if is_variable_font(original_name):
        return original_name, create_empty_moved_terms_dict()

    stem, suffixes = split_stem_and_suffixes(original_name)
    if not stem:
        return original_name, create_empty_moved_terms_dict()

    # Normalize parameters
    move_ahead_terms = move_ahead_terms or []
    move_behind_terms = move_behind_terms or []
    ignore_families = ignore_families or []

    # Process hyphenated filename
    if "-" in stem:
        parts = stem.rsplit("-", 1)
        if not (parts[0] and parts[1]):
            return original_name, create_empty_moved_terms_dict()

        family_part, style_part = parts
        clean_family, clean_style, moved_terms = extract_all_terms_from_parts(
            family_part,
            style_part,
            move_ahead_terms,
            move_behind_terms,
            number_ahead_prefix,
            number_behind_prefix,
            ignore_families,
            leading_zero,
        )

        if not any(moved_terms.values()):
            return original_name, create_empty_moved_terms_dict()

        all_terms = "".join(
            moved_terms["numbers_ahead"]
            + moved_terms["move_ahead"]
            + moved_terms["widths"]
            + moved_terms["move_behind"]
            + moved_terms["numbers_behind"]
        )
        final_stem = f"{clean_family}-{all_terms}{clean_style}"
        return f"{final_stem}{suffixes}", moved_terms

    # Process non-hyphenated filename
    clean_stem, _, moved_terms = extract_all_terms_from_parts(
        stem,
        "",
        move_ahead_terms,
        move_behind_terms,
        number_ahead_prefix,
        number_behind_prefix,
        ignore_families,
        leading_zero,
    )

    if any(moved_terms.values()):
        all_terms = "".join(
            moved_terms["numbers_ahead"]
            + moved_terms["move_ahead"]
            + moved_terms["widths"]
            + moved_terms["move_behind"]
            + moved_terms["numbers_behind"]
        )
        final_stem = f"{clean_stem}-{all_terms}"
        return f"{final_stem}{suffixes}", moved_terms

    return original_name, create_empty_moved_terms_dict()


# --- Analysis & Preview -------------------------------------------------------------------------


def analyze_files(
    paths: List[Path],
    recursive: bool,
    move_ahead_terms: List[str] | None = None,
    move_behind_terms: List[str] | None = None,
    number_ahead_prefix: str | None = None,
    number_behind_prefix: str | None = None,
    ignore_families: List[str] | None = None,
    leading_zero: bool = False,
) -> dict:
    """
    Analyze files to determine what terms will be affected.
    Returns dict with analysis data.
    """
    number_terms_ahead_found = set()
    number_terms_behind_found = set()
    move_ahead_terms_found = set()
    move_behind_terms_found = set()
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
            file_path,
            move_ahead_terms,
            move_behind_terms,
            number_ahead_prefix,
            number_behind_prefix,
            ignore_families,
            leading_zero=leading_zero,
        )
        # Only include files where the name actually changes
        if decision.has_changes():
            files_with_changes.append((file_path, decision))
            number_terms_ahead_found.update(decision.moved_numbers_ahead)
            number_terms_behind_found.update(decision.moved_numbers_behind)
            move_ahead_terms_found.update(decision.moved_ahead)
            move_behind_terms_found.update(decision.moved_behind)
            width_terms_found.update(decision.moved_widths)

    # Sort files_with_changes alphabetically by filename
    files_with_changes.sort(key=lambda x: x[0].name.lower())

    return {
        "total_files": len(font_files),
        "files_with_changes": files_with_changes,
        "number_terms_ahead_found": sorted(number_terms_ahead_found),
        "number_terms_behind_found": sorted(number_terms_behind_found),
        "move_ahead_terms_found": sorted(move_ahead_terms_found),
        "move_behind_terms_found": sorted(move_behind_terms_found),
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
    if analysis.get("number_terms_ahead_found"):
        cs.emit(
            f"{cs.indent(1)}Number terms to move ahead: {', '.join(analysis['number_terms_ahead_found'])}"
        )
    if analysis.get("number_terms_behind_found"):
        cs.emit(
            f"{cs.indent(1)}Number terms to move behind: {', '.join(analysis['number_terms_behind_found'])}"
        )
    if analysis["move_ahead_terms_found"]:
        cs.emit(
            f"{cs.indent(1)}Move-ahead terms to reorder: {', '.join(analysis['move_ahead_terms_found'])}"
        )
    if analysis["width_terms_found"]:
        cs.emit(
            f"{cs.indent(1)}Width terms to reorder: {', '.join(analysis['width_terms_found'])}"
        )
    if analysis.get("move_behind_terms_found"):
        cs.emit(
            f"{cs.indent(1)}Move-behind terms to reorder: {', '.join(analysis['move_behind_terms_found'])}"
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
                    if decision.moved_numbers_ahead:
                        moved_parts.append(
                            f"#↑: {', '.join(decision.moved_numbers_ahead)}"
                        )
                    if decision.moved_ahead:
                        moved_parts.append(f"↑: {', '.join(decision.moved_ahead)}")
                    if decision.moved_widths:
                        moved_parts.append(f"Width: {', '.join(decision.moved_widths)}")
                    if decision.moved_behind:
                        moved_parts.append(f"↓: {', '.join(decision.moved_behind)}")
                    if decision.moved_numbers_behind:
                        moved_parts.append(
                            f"#↓: {', '.join(decision.moved_numbers_behind)}"
                        )
                    moved_info = "; ".join(moved_parts)

                    table.add_row(before_highlighted, after_highlighted, moved_info)

                console = cs.get_console()
                console.print(table)
        else:
            # Fallback for non-Rich environments
            for file_path, decision in analysis["files_with_changes"]:
                moved_parts = []
                if decision.moved_numbers_ahead:
                    moved_parts.append(
                        f"numbers-ahead: {', '.join(decision.moved_numbers_ahead)}"
                    )
                if decision.moved_ahead:
                    moved_parts.append(f"move-ahead: {', '.join(decision.moved_ahead)}")
                if decision.moved_widths:
                    moved_parts.append(f"widths: {', '.join(decision.moved_widths)}")
                if decision.moved_behind:
                    moved_parts.append(
                        f"move-behind: {', '.join(decision.moved_behind)}"
                    )
                if decision.moved_numbers_behind:
                    moved_parts.append(
                        f"numbers-behind: {', '.join(decision.moved_numbers_behind)}"
                    )
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
    moved_numbers_ahead: List[str]
    moved_numbers_behind: List[str]
    moved_ahead: List[str]
    moved_widths: List[str]
    moved_behind: List[str]
    destination: Path

    def all_moved_terms(self) -> List[str]:
        """Return all moved terms in order: numbers_ahead, move-ahead, widths, move-behind, numbers_behind."""
        return (
            self.moved_numbers_ahead
            + self.moved_ahead
            + self.moved_widths
            + self.moved_behind
            + self.moved_numbers_behind
        )

    def has_changes(self) -> bool:
        """Check if any terms were moved AND the name actually changed."""
        return bool(self.all_moved_terms()) and self.new_name != self.old_name


def compute_rename(
    file_path: Path,
    move_ahead_terms: List[str] | None = None,
    move_behind_terms: List[str] | None = None,
    number_ahead_prefix: str | None = None,
    number_behind_prefix: str | None = None,
    ignore_families: List[str] | None = None,
    leading_zero: bool = False,
) -> RenameDecision:
    """
    Compute the rename decision for a file with optional term reordering.
    """
    old_name = file_path.name
    new_name, moved_terms = build_new_filename(
        old_name,
        move_ahead_terms,
        move_behind_terms,
        number_ahead_prefix,
        number_behind_prefix,
        ignore_families,
        leading_zero=leading_zero,
    )

    return RenameDecision(
        old_name=old_name,
        new_name=new_name,
        moved_numbers_ahead=moved_terms["numbers_ahead"],
        moved_numbers_behind=moved_terms["numbers_behind"],
        moved_ahead=moved_terms["move_ahead"],
        moved_widths=moved_terms["widths"],
        moved_behind=moved_terms["move_behind"],
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
    move_behind_terms: List[str] | None = None,
    number_ahead_prefix: str | None = None,
    number_behind_prefix: str | None = None,
    ignore_families: List[str] | None = None,
    leading_zero: bool = False,
) -> Tuple[bool, str | None]:
    """Perform the actual file rename operation."""
    decision = compute_rename(
        file_path,
        move_ahead_terms,
        move_behind_terms,
        number_ahead_prefix,
        number_behind_prefix,
        ignore_families,
        leading_zero=leading_zero,
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
    if decision.moved_numbers_ahead:
        moved_parts.append(f"#↑: {', '.join(decision.moved_numbers_ahead)}")
    if decision.moved_ahead:
        moved_parts.append(f"↑: {', '.join(decision.moved_ahead)}")
    if decision.moved_widths:
        moved_parts.append(f"Width: {', '.join(decision.moved_widths)}")
    if decision.moved_behind:
        moved_parts.append(f"↓: {', '.join(decision.moved_behind)}")
    if decision.moved_numbers_behind:
        moved_parts.append(f"#↓: {', '.join(decision.moved_numbers_behind)}")
    terms_info = f" ({'; '.join(moved_parts)})" if moved_parts else ""

    # Use same StatusIndicator for both dry-run and normal mode
    # DRY prefix will be added automatically when dry_run=True
    cs.StatusIndicator("updated", dry_run=dry_run).add_message(
        "rename" + terms_info
    ).add_values(old_value=decision.old_name, new_value=destination.name).emit()

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
        "-ma",
        "--move-ahead",
        action="append",
        default=[],
        metavar="TERM",
        help="Move specific term(s) ahead of width terms (repeatable, supports comma-separated). Example: -ma Alt reorders 'CondensedAlt' -> 'AltCondensed'",
    )
    parser.add_argument(
        "-mb",
        "--move-behind",
        action="append",
        default=[],
        metavar="TERM",
        help="Move specific term(s) behind width terms (repeatable, supports comma-separated). Example: -mb Curly reorders 'CurlyCondensed' -> 'CondensedCurly'",
    )
    parser.add_argument(
        "-ma-num",
        "--move-ahead-number",
        nargs="?",
        const="",
        metavar="TERM",
        help="Move number terms before width terms. With TERM: matches prefix+digits (e.g., 'No87') or digits+prefix (e.g., '87No'). Without TERM: extracts all standalone numbers. Examples: --move-ahead-number No (moves 'No87', '87No' before width), --move-ahead-number (moves all numbers before width)",
    )
    parser.add_argument(
        "-mb-num",
        "--move-behind-number",
        nargs="?",
        const="",
        metavar="TERM",
        help="Move number terms after width terms. With TERM: matches prefix+digits (e.g., 'No87') or digits+prefix (e.g., '87No'). Without TERM: extracts all standalone numbers. Examples: --move-behind-number No (moves 'No87', '87No' after width), --move-behind-number (moves all numbers after width)",
    )
    parser.add_argument(
        "--do-not-move",
        action="append",
        default=[],
        metavar="PATTERN",
        help="Don't extract numbers from family names matching these literal patterns (repeatable, supports comma-separated)",
    )
    parser.add_argument(
        "-0",
        "--leading-zero",
        action="store_true",
        help="Add leading zero to single-digit numbers (1-9) when using --move-ahead-number or --move-behind-number. Example: '1' -> '01', 'No1' -> 'No01', '1No' -> '01No'",
    )
    return parser.parse_args(argv)


def load_user_width_terms(args: argparse.Namespace) -> None:
    """Add user-defined width terms to the extractor."""
    for term in args.add_width or []:
        if term.strip():
            WIDTH_EXTRACTOR.all_terms.add(term.strip())


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
    move_behind_terms = parse_comma_separated_list(args.move_behind)
    ignore_families = parse_comma_separated_list(args.do_not_move)

    # Validate that both number flags aren't used together
    if args.move_ahead_number is not None and args.move_behind_number is not None:
        cs.StatusIndicator("error").with_explanation(
            "--move-ahead-number and --move-behind-number cannot be used together"
        ).emit()
        return 1

    # Determine which number flag was used and get the prefix
    number_ahead_prefix = None
    number_behind_prefix = None

    if args.move_ahead_number is not None:
        number_ahead_prefix = args.move_ahead_number
        # If --move-ahead-number was used without a value but followed by a path,
        # argparse may have consumed the first path as the value. Detect and fix this.
        if number_ahead_prefix and (
            number_ahead_prefix.startswith("/")
            or number_ahead_prefix.startswith("./")
            or ("/" in number_ahead_prefix and len(number_ahead_prefix) > 3)
        ):
            # This looks like a path, not a term - treat as empty string and add to paths
            args.paths.insert(0, number_ahead_prefix)
            number_ahead_prefix = ""

    if args.move_behind_number is not None:
        number_behind_prefix = args.move_behind_number
        # If --move-behind-number was used without a value but followed by a path,
        # argparse may have consumed the first path as the value. Detect and fix this.
        if number_behind_prefix and (
            number_behind_prefix.startswith("/")
            or number_behind_prefix.startswith("./")
            or ("/" in number_behind_prefix and len(number_behind_prefix) > 3)
        ):
            # This looks like a path, not a term - treat as empty string and add to paths
            args.paths.insert(0, number_behind_prefix)
            number_behind_prefix = ""

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
            move_behind_terms,
            number_ahead_prefix,
            number_behind_prefix,
            ignore_families,
            leading_zero=args.leading_zero,
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
                move_behind_terms=move_behind_terms,
                number_ahead_prefix=number_ahead_prefix,
                number_behind_prefix=number_behind_prefix,
                ignore_families=ignore_families,
                leading_zero=args.leading_zero,
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
