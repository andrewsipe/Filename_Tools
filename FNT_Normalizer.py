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
    " ": "",  # remove spaces
    "emib": "emib",
    "emil": "emil",
    "emit": "emit",
    "trab": "trab",
    "tral": "tral",
    "trat": "trat",
    "smallcaps": "Smallcaps",
    "typeface": "",
}

# Case-sensitive term normalization (with delimiters)
CASE_SENSITIVE_DICT: dict[str, str] = {
    "It.": "Italic.",
    "Ita.": "Italic.",
    "Obl.": "Oblique.",
    "Obliq.": "Oblique.",
    "SC.": "Smallcaps.",
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

    if not replaced_terms:
        return filename, []

    # Reconstruct full filename
    normalized_name = f"{result}{suffixes}"
    return normalized_name, replaced_terms


# --- Analysis & Preview -------------------------------------------------------------------------


def analyze_files(
    paths: List[Path],
    recursive: bool,
) -> dict:
    """
    Analyze files to determine what terms will be normalized.
    Returns dict with analysis data.
    """
    terms_found = set()
    files_with_changes = []

    # Collect all font files using the file collector
    font_files = collector.collect_font_files(
        paths=[str(p) for p in paths], recursive=recursive
    )

    # Analyze each file
    for file_path_str in font_files:
        file_path = Path(file_path_str)
        old_name = file_path.name
        new_name, replaced_terms = normalize_filename(old_name)

        # Only include files where the name actually changes
        if replaced_terms and new_name != old_name:
            files_with_changes.append((file_path, old_name, new_name, replaced_terms))
            terms_found.update(replaced_terms)

    # Sort files_with_changes alphabetically by filename
    files_with_changes.sort(key=lambda x: x[0].name.lower())

    return {
        "total_files": len(font_files),
        "files_with_changes": files_with_changes,
        "terms_found": sorted(terms_found),
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

                for file_path, old_name, new_name, replaced_terms in analysis[
                    "files_with_changes"
                ]:
                    # Build highlighted before/after strings
                    before_highlighted = highlight_terms_in_filename(
                        old_name, replaced_terms, style="before"
                    )
                    after_highlighted = highlight_terms_in_filename(
                        new_name, replaced_terms, style="after"
                    )

                    terms_info = ", ".join(replaced_terms)

                    table.add_row(before_highlighted, after_highlighted, terms_info)

                console = cs.get_console()
                console.print(table)
        else:
            # Fallback for non-Rich environments
            for file_path, old_name, new_name, replaced_terms in analysis[
                "files_with_changes"
            ]:
                terms_info = f" (normalized: {', '.join(replaced_terms)})"
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


def compute_rename(file_path: Path) -> RenameDecision:
    """Compute the rename decision for a file."""
    old_name = file_path.name
    new_name, replaced_terms = normalize_filename(old_name)

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
) -> Tuple[bool, str | None]:
    """Perform the actual file rename operation."""
    decision = compute_rename(file_path)
    if decision.new_name == decision.old_name:
        if verbose:
            cs.StatusIndicator("unchanged").add_file(str(file_path)).emit()
        return False, None

    destination = decision.destination

    # Simple conflict handling: skip if destination exists and is a different file
    # On case-insensitive filesystems (macOS), same file with different case will exist()
    # So we check if it's actually a different file (case-insensitively different name)
    if str(destination).lower() != str(file_path).lower() and destination.exists():
        if verbose:
            cs.StatusIndicator("warning").add_file(decision.new_name).with_explanation(
                "exists, skipping"
            ).emit()
        return False, None

    # Build info string for replaced terms
    terms_info = (
        f" (normalized: {', '.join(decision.replaced_terms)})"
        if decision.replaced_terms
        else ""
    )

    if dry_run:
        cs.StatusIndicator("info", dry_run=True).add_message(
            f"DRY-RUN normalize{terms_info}"
        ).add_values(old_value=decision.old_name, new_value=destination.name).emit()
        return True, None

    try:
        file_path.rename(destination)
        cs.StatusIndicator("updated").add_message("normalize" + terms_info).add_values(
            old_value=decision.old_name, new_value=destination.name
        ).emit()
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
    return parser.parse_args(argv)


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
        )

        if analysis["files_with_changes"]:
            show_preflight_preview(analysis)

            # Ask for confirmation unless dry-run
            if not args.dry_run:
                if not cs.prompt_confirm(
                    "Ready to normalize terms in filenames",
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
