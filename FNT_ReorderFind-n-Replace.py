#!/usr/bin/env python3
"""
Find and replace terms in font filenames by reordering them relative to hyphens.

Focus: Move specified terms to before or after the hyphen.

Logic:
1. Check for exactly one hyphen - skip if zero or multiple hyphens.
2. Split filename stem by the hyphen into family and style parts.
3. Find all matching terms in both parts using case-sensitive whole-word matching.
4. Remove terms from original positions.
5. Sort prehyphen terms alphanumerically, sort posthyphen terms alphanumerically.
6. Rebuild as: PreHyphenTerms-FamilyName-StylePart-PostHyphenTerms
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Set, Tuple

import core.core_console_styles as cs
import core.core_file_collector as collector


# --- Helpers ------------------------------------------------------------------------------------


def split_stem_and_suffixes(filename: str) -> Tuple[str, str]:
    """Split filename into stem and file extensions."""
    path = Path(filename)
    suffixes = "".join(path.suffixes)
    if not suffixes:
        return filename, ""
    return path.name[: -len(suffixes)], suffixes


# --- Term Extraction Logic ----------------------------------------------------------------------


def extract_terms(text: str, terms: Set[str]) -> Tuple[str, List[str]]:
    """
    Extract terms from text using case-sensitive whole-word matching.
    Returns (text_without_terms, list_of_terms_found).

    Finds all occurrences of each term, handles overlaps by giving precedence
    to longer terms, and returns terms in order of appearance.
    """
    if not text or not terms:
        return text, []

    # Track which parts of the string have been claimed by terms
    claimed_ranges = []
    term_matches = []  # (start, end, term) tuples

    # Sort terms by length (longest first) to handle overlaps correctly
    sorted_terms = sorted(terms, key=len, reverse=True)

    # Build regex patterns for whole-word matching (case-sensitive)
    for term in sorted_terms:
        # Escape special regex characters and use word boundaries
        pattern = re.compile(r"\b" + re.escape(term) + r"\b")
        start_pos = 0

        # Find all occurrences of this term
        while True:
            match = pattern.search(text, start_pos)
            if not match:
                break

            start, end = match.span()

            # Check if this range overlaps with any already claimed range
            overlaps = any(
                not (end <= claimed_start or start >= claimed_end)
                for claimed_start, claimed_end in claimed_ranges
            )

            if not overlaps:
                term_matches.append((start, end, term))
                claimed_ranges.append((start, end))

            start_pos = end

    # If no terms found, return unchanged
    if not term_matches:
        return text, []

    # Sort matches by position in string to maintain order
    term_matches.sort(key=lambda x: x[0])

    # Extract terms in order they appear
    terms_found = [term for _, _, term in term_matches]

    # Build the remaining text by skipping the claimed ranges
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

    remaining_text = "".join(remaining_parts)

    return remaining_text, terms_found


# --- Filename Rebuilding Logic ------------------------------------------------------------------


def build_new_filename(
    original_name: str, prehyphen_terms: Set[str], posthyphen_terms: Set[str]
) -> Tuple[str, List[str]]:
    """
    Process a filename and reorder terms relative to the hyphen.

    Only processes files with exactly one hyphen. Returns unchanged if:
    - No hyphen exists
    - Multiple hyphens exist
    - No matching terms found

    Returns (new_filename, list_of_all_moved_terms).
    """
    stem, suffixes = split_stem_and_suffixes(original_name)
    if not stem:
        return original_name, []

    # Count hyphens - must be exactly one
    hyphen_count = stem.count("-")
    if hyphen_count != 1:
        return original_name, []

    # Split on the hyphen
    parts = stem.split("-", 1)
    if not parts[0] or not parts[1]:
        return original_name, []

    family_part, style_part = parts[0], parts[1]

    # Extract prehyphen and posthyphen terms from both parts
    # Extract sequentially: first prehyphen terms, then posthyphen from cleaned text
    clean_family, family_prehyphen = extract_terms(family_part, prehyphen_terms)
    clean_family, family_posthyphen = extract_terms(clean_family, posthyphen_terms)

    clean_style, style_prehyphen = extract_terms(style_part, prehyphen_terms)
    clean_style, style_posthyphen = extract_terms(clean_style, posthyphen_terms)

    # Combine all found terms
    all_prehyphen = family_prehyphen + style_prehyphen
    all_posthyphen = family_posthyphen + style_posthyphen

    # If no terms found at all, return unchanged
    if not all_prehyphen and not all_posthyphen:
        return original_name, []

    # Sort terms alphanumerically within each group
    all_prehyphen_sorted = sorted(set(all_prehyphen))
    all_posthyphen_sorted = sorted(set(all_posthyphen))

    # Build final name: CleanFamily-PreHyphenTerms-PostHyphenTermsCleanStyle
    # Prehyphen terms go immediately after the hyphen (before the style)
    # Posthyphen terms go after the hyphen, before the remaining style

    # Build the part after the hyphen
    after_hyphen_parts = []
    if all_prehyphen_sorted:
        after_hyphen_parts.append("".join(all_prehyphen_sorted))
    if all_posthyphen_sorted:
        after_hyphen_parts.append("".join(all_posthyphen_sorted))

    # Combine all parts after hyphen
    after_hyphen = "".join(after_hyphen_parts) + clean_style

    # Build final stem: CleanFamily-AfterHyphenParts
    final_stem = f"{clean_family}-{after_hyphen}" if after_hyphen else clean_family
    final_name = f"{final_stem}{suffixes}"

    # Combine all moved terms for reporting
    all_moved_terms = all_prehyphen_sorted + all_posthyphen_sorted

    return final_name, all_moved_terms


# --- Analysis & Preview -------------------------------------------------------------------------


@dataclass
class RenameDecision:
    old_name: str
    new_name: str
    moved_terms: List[str]
    destination: Path


def compute_rename(
    file_path: Path, prehyphen_terms: Set[str], posthyphen_terms: Set[str]
) -> RenameDecision:
    """Compute the rename decision for a file."""
    old_name, (new_name, moved_terms) = (
        file_path.name,
        build_new_filename(file_path.name, prehyphen_terms, posthyphen_terms),
    )
    return RenameDecision(
        old_name=old_name,
        new_name=new_name,
        moved_terms=moved_terms,
        destination=file_path.with_name(new_name),
    )


def analyze_files(
    paths: List[Path],
    recursive: bool,
    prehyphen_terms: Set[str],
    posthyphen_terms: Set[str],
) -> dict:
    """
    Analyze files to determine what terms will be reordered.
    Returns dict with analysis data.
    """
    terms_found = set()
    files_with_changes = []
    files_skipped = []

    # Collect all font files using the file collector
    font_files = collector.collect_font_files(
        paths=[str(p) for p in paths], recursive=recursive
    )

    # Analyze each file
    for file_path_str in font_files:
        file_path = Path(file_path_str)
        decision = compute_rename(file_path, prehyphen_terms, posthyphen_terms)

        # Check if file was skipped (no/multiple hyphens) - new_name == old_name but no terms moved
        stem, _ = split_stem_and_suffixes(file_path.name)
        hyphen_count = stem.count("-") if stem else 0
        if hyphen_count != 1 and decision.new_name == decision.old_name:
            files_skipped.append(file_path)
        # Only include files where the name actually changes
        elif decision.moved_terms and decision.new_name != decision.old_name:
            files_with_changes.append((file_path, decision))
            terms_found.update(decision.moved_terms)

    # Sort files_with_changes alphabetically by filename
    files_with_changes.sort(key=lambda x: x[0].name.lower())

    return {
        "total_files": len(font_files),
        "files_with_changes": files_with_changes,
        "files_skipped": files_skipped,
        "terms_found": sorted(terms_found),
    }


def highlight_terms_in_filename(
    filename: str, terms: List[str], style: str = "before", mark_hyphen: bool = False
) -> str:
    """
    Highlight terms in a filename with the specified style.

    Args:
        filename: The filename to process
        terms: List of terms to highlight
        style: "before" (turquoise) or "after" (magenta)
        mark_hyphen: If True, also highlight the hyphen (for after style)
    """
    if not cs.RICH_AVAILABLE or not terms:
        return filename

    result = filename

    # For "after" style, highlight hyphens if needed
    if mark_hyphen and style == "after":
        # Highlight the first hyphen (after prehyphen terms)
        result = result.replace("-", "[value.after]-[/value.after]", 1)

    # Highlight each term (all occurrences)
    for term in terms:
        if style == "before":
            # Highlight all occurrences in the "before" filename
            result = result.replace(term, f"[value.before]{term}[/value.before]")
        else:  # after
            # Highlight all occurrences in the "after" filename
            result = result.replace(term, f"[value.after]{term}[/value.after]")

    return result


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
    if analysis["files_skipped"]:
        cs.emit(
            f"{cs.indent(1)}Files skipped (no/multiple hyphens): {cs.fmt_count(len(analysis['files_skipped']))}"
        )

    # Show terms that will be addressed
    if analysis["terms_found"]:
        cs.emit(f"{cs.indent(1)}Terms to reorder: {', '.join(analysis['terms_found'])}")

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
                    # Build highlighted before/after strings
                    before_highlighted = highlight_terms_in_filename(
                        decision.old_name, decision.moved_terms, style="before"
                    )
                    after_highlighted = highlight_terms_in_filename(
                        decision.new_name,
                        decision.moved_terms,
                        style="after",
                        mark_hyphen=True,
                    )

                    terms_info = ", ".join(decision.moved_terms)
                    table.add_row(before_highlighted, after_highlighted, terms_info)

                console = cs.get_console()
                console.print(table)
        else:
            # Fallback for non-Rich environments
            for file_path, decision in analysis["files_with_changes"]:
                terms_info = f" (moved: {', '.join(decision.moved_terms)})"
                cs.emit(
                    f"{cs.indent(1)}{decision.old_name} -> {decision.new_name}{terms_info}"
                )

    cs.emit("")


# --- Main Processing Logic ----------------------------------------------------------------------


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
    prehyphen_terms: Set[str],
    posthyphen_terms: Set[str],
) -> Tuple[bool, str | None]:
    """Perform the actual file rename operation."""
    decision = compute_rename(file_path, prehyphen_terms, posthyphen_terms)

    # Check if file should be skipped (no/multiple hyphens)
    stem, _ = split_stem_and_suffixes(file_path.name)
    hyphen_count = stem.count("-") if stem else 0
    if hyphen_count != 1:
        if verbose:
            reason = "no hyphen" if hyphen_count == 0 else f"{hyphen_count} hyphens"
            cs.StatusIndicator("skipped").add_file(str(file_path)).with_explanation(
                f"skipped ({reason}, need exactly one)"
            ).emit()
        return False, None

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
        terms_info = (
            f" (moved: {', '.join(decision.moved_terms)})"
            if decision.moved_terms
            else ""
        )
        cs.StatusIndicator("info", dry_run=True).add_message(
            f"DRY-RUN rename{terms_info}"
        ).add_values(old_value=decision.old_name, new_value=destination.name).emit()
        return True, None

    try:
        file_path.rename(destination)
        terms_info = (
            f" (moved: {', '.join(decision.moved_terms)})"
            if decision.moved_terms
            else ""
        )
        cs.StatusIndicator("updated").add_message("rename" + terms_info).add_values(
            old_value=decision.old_name, new_value=destination.name
        ).emit()
        return True, None
    except Exception as exc:
        return False, f"rename failed for {file_path}: {exc}"


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Find and replace terms in font filenames by reordering them relative to hyphens."
    )
    parser.add_argument(
        "paths", nargs="+", help="File(s) and/or director(y/ies) to process"
    )
    parser.add_argument(
        "--prehyphen",
        action="append",
        default=[],
        metavar="TERM",
        help="Term to move before hyphen (can be specified multiple times, case-sensitive)",
    )
    parser.add_argument(
        "--posthyphen",
        action="append",
        default=[],
        metavar="TERM",
        help="Term to move after hyphen (can be specified multiple times, case-sensitive)",
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
        "-v", "--verbose", action="store_true", help="Show unchanged and skipped files"
    )
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="Skip preflight preview and proceed directly",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str]) -> int:
    """Main entry point."""
    args = parse_args(argv)

    # Build term sets from arguments
    prehyphen_terms = set(args.prehyphen or [])
    posthyphen_terms = set(args.posthyphen or [])

    if not prehyphen_terms and not posthyphen_terms:
        cs.StatusIndicator("error").with_explanation(
            "At least one --prehyphen or --posthyphen term must be specified"
        ).emit()
        return 1

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
            path_objects, args.recursive, prehyphen_terms, posthyphen_terms
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
                prehyphen_terms=prehyphen_terms,
                posthyphen_terms=posthyphen_terms,
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
