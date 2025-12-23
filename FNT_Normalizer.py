#!/usr/bin/env python3
"""
Normalize specific terms in font filenames using case-insensitive find-and-replace.

Focus: Replace case variations of specific terms with their normalized forms.

Logic:
1. For each term in the normalization dictionary, find the first case-insensitive match.
2. Replace the matched term with its normalized form.
3. Process all dictionary terms for each filename.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

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


# --- Normalization Dictionary ---------------------------------------------------------------------

# Case-insensitive term normalization
NORMALIZATION_DICT: dict[str, str] = {
    "emib": "emib",
    "emil": "emil",
    "emit": "emit",
    "trab": "trab",
    "tral": "tral",
    "trat": "trat",
    "trah": "trah",
    "smallcaps": "Smallcaps",
    "typeface": "",
}

# Case-sensitive term normalization (with delimiters)
CASE_SENSITIVE_DICT: dict[str, str] = {
    "It.": "Italic.",
    "Ita.": "Italic.",
    "Ital.": "Italic.",
    "Itali.": "Italic.",
    "Obl.": "Oblique.",
    "Obliq.": "Oblique.",
    "SC.": "Smallcaps.",
    "Pro.": "Pro-Regular.",
    "Std.": "Std-Regular.",
    "ItalicAlt.": "AltItalic.",
    "ItalicSmallCaps": "SmallcapsItalic",
    "ItalicSmallcaps": "SmallcapsItalic",
}


# --- Helpers ------------------------------------------------------------------------------------


def split_stem_and_suffixes(filename: str) -> Tuple[str, str]:
    """Split filename into stem and file extensions."""
    path = Path(filename)
    suffix = path.suffix
    if not suffix:
        return filename, ""
    return path.stem, suffix


def is_variable_font(filename: str) -> bool:
    """Check if filename indicates a variable font (case-insensitive)."""
    stem, _ = split_stem_and_suffixes(filename)
    stem_lower = stem.lower()
    # Use word boundaries to avoid false matches (e.g., "Nova" containing "var")
    patterns = [r"\bvariable\b", r"\bvar\b", r"\bvf\b"]
    return any(re.search(pattern, stem_lower) for pattern in patterns)


def remove_spaces(text: str) -> str:
    """Remove all space characters from the given text."""
    return text.replace(" ", "")


def normalize_hyphens(text: str) -> str:
    """
    Collapse multiple hyphens into a single hyphen and strip leading/trailing hyphens.
    """
    collapsed = re.sub(r"-{2,}", "-", text)
    return collapsed.strip("-")


def remove_counter_suffix(filename: str) -> Tuple[str, bool]:
    """
    Remove ~### duplicate counter suffix from filename.

    Pattern: ~### where ### is 3 digits (e.g., ~001, ~002)
    Returns: (filename_without_counter, had_counter)

    Example:
        FontName-Bold~001.otf -> FontName-Bold.otf
    """
    # Match pattern like: FontName-Bold~001.otf
    # Pattern matches ~ followed by 3 digits before the file extension
    pattern = r"~(\d{3})(\.[^.]+)$"
    match = re.search(pattern, filename)

    if match:
        # Remove the ~### part, keep the extension
        base_name = filename[: match.start()]
        extension = match.group(2)
        return f"{base_name}{extension}", True

    return filename, False


# --- Normalization Logic ------------------------------------------------------------------------


def normalize_filename(filename: str) -> Tuple[str, List[str]]:
    """
    Normalize terms in filename using case-insensitive matching.

    Returns (normalized_filename, list_of_terms_replaced).
    Terms are replaced in dictionary order, first occurrence only.
    """
    # Skip variable fonts
    if is_variable_font(filename):
        return filename, []

    stem, suffixes = split_stem_and_suffixes(filename)
    if not stem:
        return filename, []

    replaced_terms = []
    # For case-sensitive terms (which may include periods before extensions),
    # search in the full filename to catch terms like "It." before ".otf"
    full_name = Path(filename).name
    result = full_name

    # Process case-sensitive terms first (sorted by length, longest first)
    sorted_case_sensitive = sorted(
        CASE_SENSITIVE_DICT.items(), key=lambda x: len(x[0]), reverse=True
    )

    for find_term, replace_term in sorted_case_sensitive:
        # Case-sensitive search for the term in full filename
        pattern = re.compile(re.escape(find_term))
        match = pattern.search(result)

        if match:
            matched_text = match.group(0)
            # Skip if matched text already equals replacement (redundant)
            if matched_text == replace_term:
                continue

            # Replace the matched occurrence with normalized form
            start, end = match.span()
            # If find_term ends with period but replace_term doesn't, and we're
            # replacing before an extension, preserve the period separator
            if find_term.endswith(".") and not replace_term.endswith("."):
                # Check if next char after match is start of extension (letter after period)
                if end < len(result) and result[end : end + 1].isalnum():
                    # This looks like it might be an extension - preserve period
                    result = result[:start] + replace_term + "." + result[end:]
                else:
                    result = result[:start] + replace_term + result[end:]
            else:
                result = result[:start] + replace_term + result[end:]
            replaced_terms.append(find_term)

    # After case-sensitive processing, split again to get updated stem
    # (in case the replacement changed the extension boundary)
    updated_path = Path(result)
    result = updated_path.stem
    suffixes = updated_path.suffix

    # Process case-insensitive terms (sorted by length, longest first)
    sorted_terms = sorted(
        NORMALIZATION_DICT.items(), key=lambda x: len(x[0]), reverse=True
    )

    for find_term, replace_term in sorted_terms:
        # Case-insensitive search for the term
        pattern = re.compile(re.escape(find_term), re.IGNORECASE)
        match = pattern.search(result)

        if match:
            matched_text = match.group(0)
            # Skip if matched text already equals replacement (redundant)
            if matched_text == replace_term:
                continue

            # Replace the matched occurrence with normalized form
            start, end = match.span()
            result = result[:start] + replace_term + result[end:]
            replaced_terms.append(find_term)

    # Apply global whitespace and hyphen normalization to the stem
    result = remove_spaces(result)
    result = normalize_hyphens(result)

    # If nothing changed at all, return the original filename
    if not replaced_terms and result == stem:
        return filename, []

    # Reconstruct full filename
    normalized_name = f"{result}{suffixes}"
    return normalized_name, replaced_terms


# --- Analysis & Preview -------------------------------------------------------------------------


def analyze_files(
    paths: List[Path],
    recursive: bool,
    remove_counter: bool = False,
) -> dict:
    """
    Analyze files to determine what terms will be normalized and/or counters removed.
    Returns dict with analysis data.
    """
    terms_found = set()
    files_with_changes = []
    counters_removed = 0

    # Collect all font files using the file collector
    font_files = collector.collect_font_files(
        paths=[str(p) for p in paths], recursive=recursive
    )

    # Analyze each file
    for file_path_str in font_files:
        file_path = Path(file_path_str)
        old_name = file_path.name

        # Use compute_rename to get the final name (handles both normalization and counter removal)
        decision = compute_rename(file_path, remove_counter=remove_counter)
        new_name = decision.new_name
        replaced_terms = decision.replaced_terms

        # Check if counter was removed
        had_counter = False
        if remove_counter:
            _, had_counter = remove_counter_suffix(old_name)
            if had_counter:
                counters_removed += 1

        # Include files where the name actually changes (normalization or counter removal)
        if new_name != old_name:
            files_with_changes.append(
                (file_path, old_name, new_name, replaced_terms, had_counter)
            )
            terms_found.update(replaced_terms)

    # Sort files_with_changes alphabetically by filename
    files_with_changes.sort(key=lambda x: x[0].name.lower())

    return {
        "total_files": len(font_files),
        "files_with_changes": files_with_changes,
        "terms_found": sorted(terms_found),
        "counters_removed": counters_removed,
    }


def show_preflight_preview(analysis: dict) -> None:
    """Display a preview of what will be changed."""
    cs.emit("")
    cs.StatusIndicator("info").add_message("Term Normalization Preview").emit()

    # Show statistics
    cs.emit(
        f"{cs.indent(1)}Total files scanned: {cs.fmt_count(analysis['total_files'])}"
    )
    cs.emit(
        f"{cs.indent(1)}Files requiring changes: {cs.fmt_count(len(analysis['files_with_changes']))}"
    )

    # Show terms that will be normalized
    if analysis["terms_found"]:
        cs.emit(
            f"{cs.indent(1)}Terms to normalize: {', '.join(analysis['terms_found'])}"
        )

    # Show counter removal statistics
    if analysis.get("counters_removed", 0) > 0:
        cs.emit(
            f"{cs.indent(1)}Counters to remove: {cs.fmt_count(analysis['counters_removed'])}"
        )

    cs.emit("")

    # Show ALL changes with highlighted terms
    if analysis["files_with_changes"]:
        cs.StatusIndicator("info").add_message("All changes:").emit()

        if cs.RICH_AVAILABLE:
            table = cs.create_table(show_header=True)
            if table:
                table.add_column("Original", style="lighttext", no_wrap=False)
                table.add_column("Normalized", style="lighttext", no_wrap=False)
                table.add_column("Terms", style="cyan", min_width=12)

                for file_entry in analysis["files_with_changes"]:
                    # Handle both old format (4 elements) and new format (5 elements)
                    if len(file_entry) == 5:
                        file_path, old_name, new_name, replaced_terms, had_counter = (
                            file_entry
                        )
                    else:
                        file_path, old_name, new_name, replaced_terms = file_entry
                        had_counter = False

                    # Build highlighted before/after strings
                    before_highlighted = highlight_terms_in_filename(
                        old_name, replaced_terms, style="before"
                    )
                    after_highlighted = highlight_terms_in_filename(
                        new_name, replaced_terms, style="after"
                    )

                    # Build terms info string
                    action_parts = []
                    if replaced_terms:
                        action_parts.append(", ".join(replaced_terms))
                    if had_counter:
                        action_parts.append("counter removed")
                    terms_info = ", ".join(action_parts) if action_parts else ""

                    table.add_row(before_highlighted, after_highlighted, terms_info)

                console = cs.get_console()
                console.print(table)
        else:
            # Fallback for non-Rich environments
            for file_entry in analysis["files_with_changes"]:
                # Handle both old format (4 elements) and new format (5 elements)
                if len(file_entry) == 5:
                    file_path, old_name, new_name, replaced_terms, had_counter = (
                        file_entry
                    )
                else:
                    file_path, old_name, new_name, replaced_terms = file_entry
                    had_counter = False

                action_parts = []
                if replaced_terms:
                    action_parts.append(f"normalized: {', '.join(replaced_terms)}")
                if had_counter:
                    action_parts.append("counter removed")
                terms_info = f" ({', '.join(action_parts)})" if action_parts else ""
                cs.emit(f"{cs.indent(1)}{old_name} -> {new_name}{terms_info}")

    cs.emit("")


def highlight_terms_in_filename(
    filename: str,
    terms: List[str],
    style: str = "before",
) -> str:
    """
    Highlight terms in a filename with the specified style.

    Args:
        filename: The filename to process
        terms: List of terms to highlight
        style: "before" (turquoise) or "after" (magenta)
    """
    if not cs.RICH_AVAILABLE or not terms:
        return filename

    result = filename

    # Highlight each term (case-insensitive, first occurrence only)
    for term in terms:
        pattern = re.compile(re.escape(term), re.IGNORECASE)
        match = pattern.search(result)

        if match:
            matched_text = match.group(0)
            if style == "before":
                # Highlight in the "before" filename
                result = result.replace(
                    matched_text, f"[value.before]{matched_text}[/value.before]", 1
                )
            else:  # after
                # Highlight in the "after" filename
                result = result.replace(
                    matched_text, f"[value.after]{matched_text}[/value.after]", 1
                )

    return result


# --- Main Processing Logic ----------------------------------------------------------------------


@dataclass
class RenameDecision:
    old_name: str
    new_name: str
    replaced_terms: List[str]
    destination: Path


def compute_rename(file_path: Path, remove_counter: bool = False) -> RenameDecision:
    """Compute the rename decision for a file."""
    old_name = file_path.name
    new_name, replaced_terms = normalize_filename(old_name)

    # Apply counter removal if requested (after normalization)
    if remove_counter:
        new_name, _ = remove_counter_suffix(new_name)

    return RenameDecision(
        old_name=old_name,
        new_name=new_name,
        replaced_terms=replaced_terms,
        destination=file_path.with_name(new_name),
    )


def perform_rename(
    file_path: Path,
    *,
    dry_run: bool,
    verbose: bool,
    remove_counter: bool = False,
) -> Tuple[bool, str | None]:
    """Perform the actual file rename operation."""
    decision = compute_rename(file_path, remove_counter=remove_counter)
    if decision.new_name == decision.old_name:
        if verbose:
            cs.StatusIndicator("unchanged").add_file(str(file_path)).emit()
        return False, None

    destination = decision.destination

    # Check if counter was removed (for messaging and conflict detection)
    _, had_counter = (
        remove_counter_suffix(file_path.name)
        if remove_counter
        else (file_path.name, False)
    )

    # Conflict handling: skip if destination exists and is a different file
    # On case-insensitive filesystems (macOS), same file with different case will exist()
    # So we check if it's actually a different file (case-insensitively different name)
    if str(destination).lower() != str(file_path).lower() and destination.exists():
        # Special message for counter removal conflicts
        if remove_counter and had_counter:
            cs.StatusIndicator("warning").add_file(decision.new_name).with_explanation(
                "would create duplicate, skipping (OS will handle if needed)"
            ).emit()
        else:
            cs.StatusIndicator("warning").add_file(decision.new_name).with_explanation(
                "exists, skipping"
            ).emit()
        return False, None

    # Build info string for replaced terms and counter removal
    action_parts = []
    if decision.replaced_terms:
        action_parts.append(f"normalized: {', '.join(decision.replaced_terms)}")
    if remove_counter and had_counter:
        action_parts.append("counter removed")
    terms_info = f" ({', '.join(action_parts)})" if action_parts else ""

    # Use same StatusIndicator for both dry-run and normal mode
    # DRY prefix will be added automatically when dry_run=True
    message = terms_info.strip(" ()") or "rename"
    cs.StatusIndicator("updated", dry_run=dry_run).add_message(
        message
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
        description="Normalize specific terms in font filenames using case-insensitive find-and-replace."
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
        "-v", "--verbose", action="store_true", help="Show unchanged files"
    )
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="Skip preflight preview and proceed directly",
    )
    parser.add_argument(
        "--counter-remover",
        action="store_true",
        help="Remove ~### duplicate counters from filenames (cannot be used with --recursive)",
    )
    args = parser.parse_args(argv)

    # Validate that --counter-remover is not used with --recursive
    if args.counter_remover and args.recursive:
        parser.error("--counter-remover cannot be used with --recursive")

    return args


def main(argv: Sequence[str]) -> int:
    """Main entry point."""
    args = parse_args(argv)

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
            remove_counter=args.counter_remover,
        )

        if analysis["files_with_changes"]:
            show_preflight_preview(analysis)

            # Ask for confirmation unless dry-run
            if not args.dry_run:
                action_description = "normalize terms in filenames"
                if args.counter_remover:
                    action_description = (
                        "normalize terms and remove counters in filenames"
                    )
                if not cs.prompt_confirm(
                    f"Ready to {action_description}",
                    action_prompt="Proceed with renaming?",
                    default=True,
                ):
                    cs.StatusIndicator("info").add_message("Operation cancelled").emit()
                    return 0
        else:
            cs.StatusIndicator("info").add_message(
                "No files require term normalization"
            ).emit()
            return 0

    # Process files - use same file collection as analysis
    font_files = collector.collect_font_files(
        paths=[str(p) for p in path_objects], recursive=args.recursive
    )

    total_files, changed, errors = 0, 0, 0
    for file_path_str in font_files:
        file_path = Path(file_path_str)
        total_files += 1
        did_change, error_message = perform_rename(
            file_path,
            dry_run=args.dry_run,
            verbose=args.verbose,
            remove_counter=args.counter_remover,
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
