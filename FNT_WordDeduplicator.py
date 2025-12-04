#!/usr/bin/env python3
"""
Remove duplicate/repeated words in filenames while preserving delimiters.

Example:
  Neptun-Süd-Wide-Süd-Wide-SemiBold.ttf -> Neptun-Süd-Wide-SemiBold.ttf

Rules:
- Deduplicate case-insensitively using Unicode NFC + casefold for comparison.
- Preserve original casing and delimiters of the first occurrence.
- Remove the immediate preceding hyphen/underscore/space when a duplicate token
  is removed to avoid stray separators.
- Only alphabetic segments are considered "words"; other parts are preserved.

Safety & CLI:
- Conflicts handled via --conflict (skip|unique|overwrite), default unique.
- Dry-run supported.
- Optional cleanup pass removes " (n)" suffixes when the clean name exists.
"""

from __future__ import annotations

import argparse
import re
import sys
import unicodedata
from pathlib import Path
from typing import Iterable, List, Tuple

import core.core_console_styles as cs


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


def tokenize_segments(basename: str) -> List[str]:
    """Split basename into word vs delimiter segments, preserving delimiters.

    Word segments include letters and their Unicode combining marks (Mn/Mc/Me),
    so sequences like "Su\u0308d" stay intact and don't leak marks when removed.
    """
    if not basename:
        return []

    def _is_mark(ch: str) -> bool:
        try:
            return unicodedata.category(ch).startswith("M")
        except Exception:
            return False

    def _is_word_char(ch: str) -> bool:
        return ch.isalpha() or _is_mark(ch)

    parts: List[str] = []
    buf: List[str] = []
    is_word_prev = _is_word_char(basename[0])
    for ch in basename:
        is_word = _is_word_char(ch)
        if is_word == is_word_prev:
            buf.append(ch)
        else:
            parts.append("".join(buf))
            buf = [ch]
            is_word_prev = is_word
    if buf:
        parts.append("".join(buf))
    return parts


def _is_alpha_part(text: str) -> bool:
    if not text:
        return False
    # Consider a part a word if it contains at least one letter. Combining marks
    # alone (rare at string start) are not treated as a separate word.
    return any(ch.isalpha() for ch in text)


def normalize_for_compare(token: str) -> str:
    """Normalize token for comparison: NFC + casefold."""
    try:
        return unicodedata.normalize("NFC", token).casefold()
    except Exception:
        return token.lower()


def deduplicate_words_in_basename(basename: str) -> Tuple[str, List[str]]:
    """Remove repeated alphabetic tokens, preserving first occurrence and delimiters.

    Returns (new_basename, removed_tokens_keys) where removed_tokens_keys are the
    normalized keys of tokens removed (for potential diagnostics).
    """
    parts = tokenize_segments(basename)
    if not parts:
        return basename, []

    seen: set[str] = set()
    removed: List[str] = []
    out_parts: List[str] = []

    for part in parts:
        if not _is_alpha_part(part):
            out_parts.append(part)
            continue

        key = normalize_for_compare(part)
        if key in seen:
            # Remove a simple preceding separator like '-', '_', or space if present
            if (
                out_parts
                and not _is_alpha_part(out_parts[-1])
                and all(ch in "-_ " for ch in out_parts[-1])
            ):
                out_parts.pop()
            removed.append(key)
            continue

        seen.add(key)
        out_parts.append(part)

    new_text = "".join(out_parts)
    # Collapse leftover multiple hyphens
    new_text = re.sub(r"-{2,}", "-", new_text)
    return new_text, removed


def build_new_filename(original_name: str) -> Tuple[str, bool]:
    stem, suffixes = split_stem_and_suffixes(original_name)
    new_stem, removed = deduplicate_words_in_basename(stem)
    return f"{new_stem}{suffixes}", bool(removed) and (
        f"{new_stem}{suffixes}" != original_name
    )


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
    old_name = file_path.name
    new_name, changed = build_new_filename(old_name)

    if not changed:
        if verbose:
            cs.StatusIndicator("unchanged").add_file(str(file_path)).emit()
        return False, None

    destination = file_path.with_name(new_name)
    if destination.exists():
        if conflict == "skip":
            if verbose:
                cs.StatusIndicator("warning").add_file(new_name).with_explanation(
                    "exists, skipping"
                ).emit()
            return False, None
        if conflict == "unique":
            destination = ensure_unique_destination(destination)
        # overwrite: proceed

    if dry_run:
        cs.StatusIndicator("info", dry_run=True).add_message(
            f"DRY-RUN rename in {cs.fmt_file(str(file_path.parent))}"
        ).add_values(old_value=old_name, new_value=destination.name).emit()
        return True, None

    try:
        file_path.rename(destination)
        cs.StatusIndicator("updated").add_message(
            f"in {cs.fmt_file(str(file_path.parent))}"
        ).add_values(old_value=old_name, new_value=destination.name).emit()
        return True, None
    except Exception as exc:  # noqa: BLE001
        return False, f"rename failed for {file_path}: {exc}"


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Remove duplicate/repeated words in filenames, preserving delimiters."
        )
    )
    parser.add_argument(
        "paths",
        nargs="+",
        help="File(s) and/or director(y/ies) to process",
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
        help="On name conflict: skip, create a unique name, or overwrite (default: unique)",
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
    return parser.parse_args(argv)


def main(argv: Iterable[str]) -> int:
    args = parse_args(argv)
    total_files = 0
    changed = 0
    errors = 0

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
