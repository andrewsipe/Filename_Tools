#!/usr/bin/env python3
"""
Convert spelled-out number words to integers in filenames (stem only).

Examples:
  One -> 1
  OneHundred -> 100
  OneThousand -> 1000
  Eleven -> 11
  Twelve -> 12
  HelveticaOne-OneHundred -> Helvetica1-100

Handles both CamelCase/PascalCase (OneHundred) and space-separated (One Hundred) formats.
Processes each segment independently to avoid cross-boundary combinations.

CLI:
- paths... [-r] [-n] [--conflict skip|unique|overwrite] [-v] [--cleanup|--no-cleanup]
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

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

try:
    import word2number.w2n as w2n  # type: ignore

    _W2N_AVAILABLE = True
except Exception:  # pragma: no cover - env dependent
    w2n = None  # type: ignore
    _W2N_AVAILABLE = False


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
    """Split basename into word vs delimiter segments, preserving delimiters."""
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


def _is_single_digit_number(word: str) -> bool:
    """Check if a word represents a single-digit number (0-9).

    Returns True if the word converts to a number between 0 and 9.
    """
    if not _W2N_AVAILABLE:
        return False
    try:
        number = w2n.word_to_num(word.lower())  # type: ignore
        return 0 <= number <= 9
    except Exception:
        return False


def split_camelcase(word: str) -> List[str]:
    """Split CamelCase/PascalCase word into individual words.

    Examples:
        OneHundred -> ['One', 'Hundred']
        OneThousand -> ['One', 'Thousand']
        Eleven -> ['Eleven']
        HelveticaOne -> ['Helvetica', 'One']
    """
    if not word:
        return []
    # Find boundaries: lowercase/uppercase transitions and uppercase sequences
    words: List[str] = []
    current: List[str] = []

    for i, ch in enumerate(word):
        if ch.isupper():
            # If we have accumulated lowercase letters, start a new word
            if current and current[-1].islower():
                words.append("".join(current))
                current = [ch]
            else:
                current.append(ch)
        elif ch.islower():
            current.append(ch)
        else:
            # Non-letter character - finish current word if any
            if current:
                words.append("".join(current))
                current = []

    if current:
        words.append("".join(current))

    return words if words else [word]


def find_number_sequence(
    words: List[str], start_idx: int
) -> Optional[Tuple[int, int, int]]:
    """Find the longest valid number sequence starting at start_idx.

    Returns (start, end, number) if found, None otherwise.
    end is exclusive (like slice notation).

    Important: Only accepts sequences where ALL words are part of the number.
    If adding a word doesn't change the number (e.g., "three regular" = 3, same as "three"),
    we stop at the shorter sequence to avoid consuming non-number words.
    """
    if not _W2N_AVAILABLE:
        return None

    # Try sequences of increasing length
    best_match: Optional[Tuple[int, int, int]] = None
    previous_number: Optional[int] = None

    for end_idx in range(start_idx + 1, len(words) + 1):
        sequence = words[start_idx:end_idx]
        # Join with spaces for w2n
        number_text = " ".join(w.lower() for w in sequence)

        try:
            number = w2n.word_to_num(number_text)  # type: ignore

            # If this is a longer sequence and the number hasn't changed,
            # it means the extra words are being ignored (not part of the number).
            # Stop here and use the shorter sequence.
            if previous_number is not None and number == previous_number:
                # The number didn't change when we added more words,
                # so those extra words aren't part of the number
                break

            # Store the longest valid sequence so far
            best_match = (start_idx, end_idx, number)
            previous_number = number
        except Exception:
            # Invalid sequence, stop trying longer sequences
            break

    return best_match


def convert_numbers_in_segment(
    segment: str, leading_zero: bool = False
) -> Tuple[str, List[Tuple[str, str]]]:
    """Convert spelled-out numbers in a single segment.

    Handles both CamelCase and space-separated formats.
    Processes the segment independently to avoid cross-boundary combinations.

    Returns (converted_segment, list of changes).
    """
    if not segment or not _is_alpha_part(segment):
        return segment, []

    if not _W2N_AVAILABLE:
        return segment, []

    changes: List[Tuple[str, str]] = []

    # Check if segment contains spaces (space-separated format)
    if " " in segment:
        # Space-separated format: split on spaces
        words = segment.split()
    else:
        # CamelCase format: split CamelCase into words
        words = split_camelcase(segment)

    if not words:
        return segment, []

    # Find and convert number sequences
    # Important: We must preserve ALL words, only converting number words to digits
    converted_words: List[str] = []
    i = 0
    while i < len(words):
        # Check if we have consecutive single-digit numbers (for concatenation)
        if _is_single_digit_number(words[i]):
            # Count how many consecutive single-digit numbers we have
            consecutive_count = 1
            while i + consecutive_count < len(words) and _is_single_digit_number(
                words[i + consecutive_count]
            ):
                consecutive_count += 1

            # If we have 2+ consecutive single-digit numbers, concatenate them
            if consecutive_count >= 2:
                # Convert each single-digit number individually and concatenate
                digit_strings: List[str] = []
                original_sequence_parts: List[str] = []
                for j in range(consecutive_count):
                    word = words[i + j]
                    original_sequence_parts.append(word)
                    try:
                        digit = w2n.word_to_num(word.lower())  # type: ignore
                        # Apply leading zero formatting if requested
                        if leading_zero and 1 <= digit <= 9:
                            digit_str = f"{digit:02d}"
                        else:
                            digit_str = str(digit)
                        digit_strings.append(digit_str)
                    except Exception:
                        # If conversion fails, keep the original word
                        digit_strings.append(word)

                # Concatenate all digits
                concatenated = "".join(digit_strings)
                original_sequence = (
                    "".join(original_sequence_parts)
                    if " " not in segment
                    else " ".join(original_sequence_parts)
                )
                converted_words.append(concatenated)
                if original_sequence != concatenated:
                    changes.append((original_sequence, concatenated))
                i += consecutive_count
                continue

        # Not consecutive single-digit numbers - use existing logic
        match = find_number_sequence(words, i)
        if match:
            start, end, number = match
            # Convert the number sequence to digit
            original_sequence = (
                "".join(words[start:end])
                if " " not in segment
                else " ".join(words[start:end])
            )
            # Apply leading zero formatting if requested and number is 1-9
            if leading_zero and 1 <= number <= 9:
                number_str = f"{number:02d}"  # 1 -> "01", 9 -> "09"
            else:
                number_str = str(number)  # 10 -> "10", 100 -> "100"
            converted_words.append(number_str)
            if original_sequence != number_str:
                changes.append((original_sequence, number_str))
            # Move past the converted sequence - this ensures we skip the words
            # that were part of the number but continue processing remaining words
            i = end
        else:
            # Not a number word - preserve it exactly as-is
            converted_words.append(words[i])
            i += 1

    # Verify we processed all words (safety check)
    if len(converted_words) == 0 and len(words) > 0:
        # Fallback: if something went wrong, return original
        return segment, []

    # Reconstruct segment - ensure all words are preserved
    if " " in segment:
        # Space-separated: join with spaces
        new_segment = " ".join(converted_words)
    else:
        # CamelCase: join without spaces
        # Preserve original casing for non-number words
        # Numbers are already strings, non-numbers keep their original case
        new_segment = "".join(converted_words)

    return new_segment, changes


def convert_number_words_in_basename(
    basename: str, leading_zero: bool = False
) -> Tuple[str, List[Tuple[str, str]]]:
    """Convert spelled-out numbers in basename, processing each segment independently.

    Returns (converted_basename, list of changes).
    """
    if not basename:
        return basename, []

    parts = tokenize_style_segments(basename)
    if not parts:
        return basename, []

    all_changes: List[Tuple[str, str]] = []
    new_parts: List[str] = []

    for part in parts:
        if _is_alpha_part(part):
            converted, changes = convert_numbers_in_segment(
                part, leading_zero=leading_zero
            )
            new_parts.append(converted)
            all_changes.extend(changes)
        else:
            # Delimiter - preserve as-is
            new_parts.append(part)

    new_basename = "".join(new_parts)
    return new_basename, all_changes


def build_new_filename(
    original_name: str, leading_zero: bool = False
) -> Tuple[str, List[Tuple[str, str]]]:
    """Build new filename with number words converted to integers."""
    stem, suffixes = split_stem_and_suffixes(original_name)
    new_stem, changes = convert_number_words_in_basename(
        stem, leading_zero=leading_zero
    )
    return f"{new_stem}{suffixes}", changes


# --- Analysis & Preview -------------------------------------------------------------------------


@dataclass
class RenameDecision:
    old_name: str
    new_name: str
    changes: List[Tuple[str, str]]
    destination: Path


def compute_rename(file_path: Path, leading_zero: bool = False) -> RenameDecision:
    """Compute the rename decision for a file."""
    old_name = file_path.name
    new_name, changes = build_new_filename(old_name, leading_zero=leading_zero)
    return RenameDecision(
        old_name=old_name,
        new_name=new_name,
        changes=changes,
        destination=file_path.with_name(new_name),
    )


def analyze_files(
    paths: List[Path], recursive: bool, leading_zero: bool = False
) -> dict:
    """
    Analyze files to determine what will be changed.
    Returns dict with analysis data.
    """
    files_with_changes = []
    number_conversions = set()

    # Analyze each file
    for path in paths:
        for file_path in iter_target_files(path, recursive):
            decision = compute_rename(file_path, leading_zero=leading_zero)
            # Only include files where the name actually changes
            if decision.new_name != decision.old_name:
                files_with_changes.append((file_path, decision))
                # Collect all number conversions for summary
                for old_val, new_val in decision.changes:
                    number_conversions.add(f"{old_val} → {new_val}")

    # Sort files_with_changes alphabetically by filename
    files_with_changes.sort(key=lambda x: x[0].name.lower())

    # Count total files
    total_files = sum(1 for path in paths for _ in iter_target_files(path, recursive))

    return {
        "total_files": total_files,
        "files_with_changes": files_with_changes,
        "number_conversions": sorted(number_conversions),
    }


def show_preflight_preview(analysis: dict) -> None:
    """Display a preview of what will be changed."""
    cs.emit("")
    cs.StatusIndicator("info").add_message("Number Word Conversion Preview").emit()

    # Show statistics
    cs.emit(
        f"{cs.indent(1)}Total files scanned: {cs.fmt_count(analysis['total_files'])}"
    )
    cs.emit(
        f"{cs.indent(1)}Files requiring changes: {cs.fmt_count(len(analysis['files_with_changes']))}"
    )

    # Show number conversions that will be made
    if analysis["number_conversions"]:
        cs.emit(
            f"{cs.indent(1)}Number conversions: {', '.join(analysis['number_conversions'])}"
        )

    cs.emit("")

    # Show ALL changes
    if analysis["files_with_changes"]:
        cs.StatusIndicator("info").add_message("All changes:").emit()

        if cs.RICH_AVAILABLE:
            table = cs.create_table(show_header=True)
            if table:
                table.add_column("Original", style="lighttext", no_wrap=False)
                table.add_column("New Name", style="lighttext", no_wrap=False)
                table.add_column("Conversions", style="cyan", min_width=20)

                for file_path, decision in analysis["files_with_changes"]:
                    # Build conversions info
                    conv_parts = [f"{old}→{new}" for old, new in decision.changes]
                    conv_info = ", ".join(conv_parts) if conv_parts else ""

                    table.add_row(decision.old_name, decision.new_name, conv_info)

                console = cs.get_console()
                console.print(table)
        else:
            # Fallback for non-Rich environments
            for file_path, decision in analysis["files_with_changes"]:
                conv_parts = [f"{old}→{new}" for old, new in decision.changes]
                conv_info = f" ({', '.join(conv_parts)})" if conv_parts else ""
                cs.emit(
                    f"{cs.indent(1)}{decision.old_name} → {decision.new_name}{conv_info}"
                )

    cs.emit("")


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
                    "DRY-RUN cleanup"
                ).add_values(old_value=file_path.name, new_value=clean_name).emit()
            else:
                try:
                    file_path.rename(clean_path)
                    cs.StatusIndicator("updated").add_message("cleanup").add_values(
                        old_value=file_path.name, new_value=clean_name
                    ).emit()
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
    leading_zero: bool = False,
) -> Tuple[bool, str | None]:
    original_value = file_path.name
    new_value, changes = build_new_filename(original_value, leading_zero=leading_zero)
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
            "DRY-RUN rename"
        ).add_values(old_value=original_value, new_value=destination.name).emit()
        return True, None

    try:
        file_path.rename(destination)
        cs.StatusIndicator("updated").add_message("rename").add_values(
            old_value=original_value, new_value=destination.name
        ).emit()
        return True, None
    except Exception as exc:  # noqa: BLE001
        return False, f"rename failed for {file_path}: {exc}"


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert spelled-out number words to integers in filenames (stem only)."
        )
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
        "--no-preview",
        action="store_true",
        help="Skip preflight preview and proceed directly",
    )
    parser.add_argument(
        "-0",
        "--leading-zero",
        action="store_true",
        help="Add leading zero to single-digit numbers (1-9) -> (01-09)",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str]) -> int:
    if not _W2N_AVAILABLE:
        cs.StatusIndicator("error").with_explanation(
            "word2number library not available. Install with: pip install word2number"
        ).emit()
        return 1

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
            path_objects, args.recursive, leading_zero=args.leading_zero
        )

        if analysis["files_with_changes"]:
            show_preflight_preview(analysis)

            # Ask for confirmation unless dry-run
            if not args.dry_run:
                if not cs.prompt_confirm(
                    "Ready to convert number words to integers",
                    action_prompt="Proceed with renaming?",
                    default=True,
                ):
                    cs.StatusIndicator("info").add_message("Operation cancelled").emit()
                    return 0
        else:
            cs.StatusIndicator("info").add_message(
                "No files require number word conversion"
            ).emit()
            return 0

    # Process files
    total_files = 0
    changed = 0
    errors = 0

    for path in path_objects:
        for file_path in iter_target_files(path, args.recursive):
            total_files += 1
            did_change, error_message = perform_rename(
                file_path,
                dry_run=args.dry_run,
                conflict=args.conflict,
                verbose=args.verbose,
                leading_zero=args.leading_zero,
            )
            if did_change:
                changed += 1
            if error_message is not None:
                errors += 1
                cs.StatusIndicator("error").with_explanation(error_message).emit()

    if args.cleanup:
        for path in path_objects:
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
