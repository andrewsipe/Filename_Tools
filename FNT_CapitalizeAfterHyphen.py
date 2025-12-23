#!/usr/bin/env python3
"""
Capitalize the first letter and the first letter after each hyphen/underscore/space in filenames.
Underscores and spaces are then removed; hyphens are preserved.

Examples:
  my-file.ttf          -> My-File.ttf
  my_file_name.otf     -> MyFileName.otf
  my file-name.ttf     -> MyFile-Name.ttf

By default only the basename (without extension) is transformed. Use
--include-extension to also transform the extension portion.

Safety:
- By default, on conflicts a unique name like "name (1).ext" is generated, then
  cleaned up to remove the " (n)" suffix when possible.
- Use --conflict=skip to skip conflicting files.
- Use --conflict=overwrite to overwrite existing files (use with caution).
- Use --no-cleanup to keep " (n)" suffixes when they're created.

Usage:
  python CapitalizeAfterHyphen.py PATH [PATH ...]
  python CapitalizeAfterHyphen.py DIR -r
  python CapitalizeAfterHyphen.py DIR --no-cleanup  # Keep " (n)" suffixes
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Iterable, Tuple

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


def transform_segment_capitalize_after_hyphen(text: str) -> str:
    """Return text capitalizing the first alphabetical character and those following
    '-', '_' or space. Remove '_' and space; preserve '-'.

    Examples:
      "my_file name" -> "MyFileName"
      "my-file name" -> "My-FileName"
    """
    if not text:
        return text

    out: list[str] = []
    capitalize_next = True  # Capitalize the first alphabetical character

    for ch in text:
        if ch == "-":
            out.append(ch)
            capitalize_next = True
            continue
        if ch == "_" or ch == " ":
            # Remove these delimiters but capitalize the next letter
            capitalize_next = True
            continue

        if capitalize_next and ch.isalpha():
            out.append(ch.upper())
            capitalize_next = False
        else:
            out.append(ch)

    return "".join(out)


def build_new_filename(original_name: str, include_extension: bool) -> str:
    """Construct a new filename by transforming characters after '-'.

    If include_extension is False, transforms only the stem, preserving
    the full extension (including multi-part extensions like .tar.gz).
    """
    original_path = Path(original_name)
    if include_extension:
        return transform_segment_capitalize_after_hyphen(original_path.name)

    stem = original_path.stem
    suffix = "".join(original_path.suffixes)
    new_stem = transform_segment_capitalize_after_hyphen(stem)
    return f"{new_stem}{suffix}"


def insert_before_extension(filename: str, insertion: str) -> str:
    """Insert a string before the extension(s) of the filename.

    For multi-part extensions, the insertion is added before the first dot
    of the extension chain, e.g., name.tar.gz -> name{ins}.tar.gz.
    """
    p = Path(filename)
    suffix = "".join(p.suffixes)
    if not suffix:
        return f"{filename}{insertion}"
    stem = p.name[: -len(suffix)]
    return f"{stem}{insertion}{suffix}"


def ensure_unique_destination(path: Path) -> Path:
    """Return a non-existing sibling Path by appending " (n)" before extension."""
    if not path.exists():
        return path

    counter = 1
    candidate = path
    while candidate.exists():
        candidate_name = insert_before_extension(path.name, f" ({counter})")
        candidate = path.with_name(candidate_name)
        counter += 1
    return candidate


def iter_target_files(root: Path, recursive: bool) -> Iterable[Path]:
    """Yield files under root. If root is a file, yield it. If a dir, iterate.

    Directories themselves are not renamed to avoid traversal issues.
    """

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
            for file_path in root.rglob("*"):
                if file_path.is_file() and not _is_hidden(file_path):
                    yield file_path
        else:
            for file_path in root.iterdir():
                if file_path.is_file() and not _is_hidden(file_path):
                    yield file_path


def cleanup_numbered_duplicates(
    root: Path, recursive: bool, dry_run: bool, verbose: bool
) -> None:
    """Remove ' (n)' suffixes from filenames when the clean name is available."""
    # Pattern to match " (n)" before extension
    pattern = re.compile(r"^(.+) \((\d+)\)(\..+)?$")

    for file_path in iter_target_files(root, recursive):
        match = pattern.match(file_path.name)
        if not match:
            continue

        base_name = match.group(1)
        extension = match.group(3) or ""
        clean_name = f"{base_name}{extension}"

        clean_path = file_path.with_name(clean_name)
        if not clean_path.exists():
            # Use same StatusIndicator for both dry-run and normal mode
            # DRY prefix will be added automatically when dry_run=True
            cs.StatusIndicator("updated", dry_run=dry_run).add_message("cleanup").add_values(
                old_value=file_path.name, new_value=clean_name
            ).emit()

            if dry_run:
                continue

            try:
                file_path.rename(clean_path)
            except Exception as exc:
                cs.StatusIndicator("error", dry_run=dry_run).add_file(
                    file_path.name
                ).with_explanation(
                    f"cleanup error: failed to rename {file_path.name} -> {clean_name}: {exc}"
                ).emit()
        elif verbose:
            cs.StatusIndicator("unchanged").add_file(file_path.name).with_explanation(
                "cleanup skip (target exists)"
            ).emit()


def compute_rename(file_path: Path, include_extension: bool) -> Tuple[str, str]:
    """Compute old->new name mapping (basename only), without touching disk."""
    old_name = file_path.name
    new_name = build_new_filename(old_name, include_extension)
    return old_name, new_name


def perform_rename(
    file_path: Path,
    include_extension: bool,
    dry_run: bool,
    conflict_strategy: str,
    verbose: bool,
) -> None:
    old_name, new_name = compute_rename(file_path, include_extension)

    if new_name == old_name:
        if verbose:
            cs.StatusIndicator("unchanged").add_file(str(file_path)).with_explanation(
                "skip (no change)"
            ).emit()
        return

    destination = file_path.with_name(new_name)
    if destination.exists():
        if conflict_strategy == "skip":
            cs.StatusIndicator("warning").add_file(file_path.name).with_explanation(
                f"skip (exists): {file_path.name} -> {new_name}"
            ).emit()
            return
        elif conflict_strategy == "unique":
            destination = ensure_unique_destination(destination)
        # elif conflict_strategy == "overwrite": proceed with rename (will overwrite)

    # Use same StatusIndicator for both dry-run and normal mode
    # DRY prefix will be added automatically when dry_run=True
    cs.StatusIndicator("updated", dry_run=dry_run).add_message("renamed").add_values(
        old_value=file_path.name, new_value=destination.name
    ).emit()

    if dry_run:
        return

    try:
        file_path.rename(destination)
    except Exception as exc:  # noqa: BLE001
        cs.StatusIndicator("error").add_file(str(file_path)).with_explanation(
            f"Error: failed to rename {file_path} -> {destination.name}: {exc}"
        ).emit()


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Capitalize the first letter and after each hyphen/underscore/space; remove '_' and spaces."
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
        "-e",
        "--include-extension",
        action="store_true",
        help="Also transform the extension portion",
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
        help="After renaming, remove ' (n)' suffixes when the clean name is available (default: enabled)",
    )
    parser.add_argument(
        "--no-cleanup",
        action="store_false",
        dest="cleanup",
        help="Skip the cleanup pass (keeps ' (n)' suffixes)",
    )

    # If no arguments are provided and cleanup is not explicitly disabled,
    # assume current directory for cleanup operations.
    if not argv and "--cleanup" not in argv and "--no-cleanup" not in argv:
        # This branch is tricky because if 'paths' is required, it can't be empty.
        # The user's original call had no paths, so this might be the intent.
        # However, the script is designed to *require* paths for its primary function.
        # This change would essentially make paths optional, which is not what
        # the 'required positional argument' error usually implies.
        # Reverting to original behavior to ensure 'paths' is always expected.
        pass

    return parser.parse_args(argv)


def main(argv: Iterable[str]) -> int:
    args = parse_args(argv)
    any_target = False

    # Main renaming pass
    for raw_path in args.paths:
        path = Path(raw_path)
        if not path.exists():
            cs.StatusIndicator("warning").add_file(raw_path).with_explanation(
                "Warning: path not found"
            ).emit()
            continue

        for file_path in iter_target_files(path, args.recursive):
            any_target = True
            perform_rename(
                file_path=file_path,
                include_extension=args.include_extension,
                dry_run=args.dry_run,
                conflict_strategy=args.conflict,
                verbose=args.verbose,
            )

    # Cleanup pass to remove " (n)" suffixes
    if args.cleanup and any_target:
        if args.verbose:
            print("\n--- Cleanup pass ---")
        for raw_path in args.paths:
            path = Path(raw_path)
            if path.exists():
                cleanup_numbered_duplicates(
                    path, args.recursive, args.dry_run, args.verbose
                )

    if not any_target:
        cs.StatusIndicator("info").add_message("nothing to do").emit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
