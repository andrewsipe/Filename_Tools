#!/usr/bin/env python3
"""
Normalize hyphen placement around known style terms and remove hyphen after style prefixes.

Focus: Hyphenation consistency only.

Rules:
- Left-hyphen terms (e.g., Condensed): enforce "-Term" when adjacent (FooTerm-Bar -> Foo-TermBar).
- Right-hyphen terms (e.g., Display): enforce "Term-" when adjacent (Foo-Term -> FooTerm-).
- Compound prefixes (e.g., Semi-, Extra-, Ultra-, Super-, X-, Demi-): remove hyphen immediately after
  when followed by a letter (Semi-Bold -> SemiBold).

CLI:
- paths... [-r] [-n] [--conflict skip|unique|overwrite] [-v] [--cleanup|--no-cleanup]
- Optional dictionary tuning: --rules-file, --add-hyphen-left TERM, --add-hyphen-right TERM
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import core.core_console_styles as cs


HYPHEN_LEFT_TERMS: set[str] = {
    "Condensed",
    "SemiCondensed",
    "ExtraCondensed",
    "UltraCondensed",
    "Compressed",
    "SemiCompressed",
    "ExtraCompressed",
    "UltraCompressed",
    "Compact",
    "SemiCompact",
    "ExtraCompact",
    "UltraCompact",
    "Narrow",
    "SemiNarrow",
    "ExtraNarrow",
    "UltraNarrow",
    "Expanded",
    "SemiExpanded",
    "ExtraExpanded",
    "UltraExpanded",
    "Extended",
    "SemiExtended",
    "ExtraExtended",
    "UltraExtended",
    "Wide",
    "SemiWide",
    "ExtraWide",
    "UltraWide",
    "Variable",
}

HYPHEN_RIGHT_TERMS: set[str] = {
    "Display",
    "Text",
    "Caption",
    "Subhead",
    "Headline",
    "Title",
    "Poster",
    "Deck",
    "Micro",
    "Round",
    "Mono",
    "Sans",
    "Slab",
    "Serif",
}

COMPOUND_PREFIXES: set[str] = {"Semi", "Extra", "Demi", "X", "Ultra", "Super"}


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


def apply_hyphen_placement_rules(text: str) -> Tuple[str, List[Tuple[str, str]]]:
    if not text:
        return text, []
    changes: List[Tuple[str, str]] = []
    for term in sorted(HYPHEN_LEFT_TERMS, key=len, reverse=True):
        before = text
        text = text.replace(f"{term}-", f"-{term}")
        if text != before:
            changes.append((before, text))
        i = 0
        while True:
            idx = text.find(term, i)
            if idx == -1:
                break
            left_ok = idx > 0 and text[idx - 1] == "-"
            if not left_ok and idx > 0 and text[idx - 1].isalpha():
                new_text = text[:idx] + "-" + text[idx:]
                changes.append((text, new_text))
                text = new_text
                i = idx + len(term) + 1
            else:
                i = idx + len(term)
        before = text
        text = text.replace(f"-{term}-", f"-{term}")
        if text != before:
            changes.append((before, text))

    for term in sorted(HYPHEN_RIGHT_TERMS, key=len, reverse=True):
        before = text
        text = text.replace(f"-{term}", f"{term}-")
        if text != before:
            changes.append((before, text))
        i = 0
        while True:
            idx = text.find(term, i)
            if idx == -1:
                break
            end = idx + len(term)
            right_ok = end < len(text) and text[end] == "-"
            if not right_ok and end < len(text) and text[end].isalpha():
                new_text = text[:end] + "-" + text[end:]
                changes.append((text, new_text))
                text = new_text
                i = end + 1
            else:
                i = end
        before = text
        text = text.replace(f"-{term}-", f"{term}-")
        if text != before:
            changes.append((before, text))

    collapsed = re.sub(r"-{2,}", "-", text)
    if collapsed != text:
        changes.append((text, collapsed))
        text = collapsed
    return text, changes


def apply_compound_prefix_unhyphen(text: str) -> Tuple[str, List[Tuple[str, str]]]:
    if not text:
        return text, []
    changes: List[Tuple[str, str]] = []
    try:
        if not COMPOUND_PREFIXES:
            return text, changes
        alt = "|".join(sorted(COMPOUND_PREFIXES, key=len, reverse=True))
        pattern = re.compile(rf"(?P<prefix>{alt})-(?=[A-Za-z])")

        def _repl(m: re.Match[str]) -> str:
            return m.group("prefix")

        before = text
        after = pattern.sub(_repl, text)
        if after != before:
            changes.append((before, after))
            text = after
    except Exception:
        return text, changes
    return text, changes


def build_new_filename(original_name: str) -> Tuple[str, List[Tuple[str, str]]]:
    stem, suffixes = split_stem_and_suffixes(original_name)
    after_hyphen, changes1 = apply_hyphen_placement_rules(stem)
    after_unhyphen, changes2 = apply_compound_prefix_unhyphen(after_hyphen)
    return f"{after_unhyphen}{suffixes}", (changes1 + changes2)


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
    new_name, changes = build_new_filename(old_name)
    if new_name == old_name:
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


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Normalize hyphen placement and remove hyphen after style prefixes in filenames."
        ),
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
        "--rules-file",
        dest="rules_file",
        help="JSON with hyphen_left, hyphen_right, prefixes",
    )
    parser.add_argument(
        "--add-hyphen-left",
        action="append",
        default=[],
        metavar="TERM",
        help="Add a left-hyphen term (e.g., -Condensed)",
    )
    parser.add_argument(
        "--add-hyphen-right",
        action="append",
        default=[],
        metavar="TERM",
        help="Add a right-hyphen term (e.g., Display-)",
    )
    return parser.parse_args(argv)


def load_user_rules(args: argparse.Namespace) -> None:
    if args.rules_file:
        import json

        try:
            with open(args.rules_file, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception as exc:  # noqa: BLE001
            cs.StatusIndicator("error").with_explanation(
                f"failed to read rules file: {exc}"
            ).emit()
            data = {}
        hy_left = data.get("hyphen_left") or data.get("hyphen-left") or []
        if isinstance(hy_left, list):
            for term in hy_left:
                if isinstance(term, str) and term:
                    HYPHEN_LEFT_TERMS.add(term)
        hy_right = data.get("hyphen_right") or data.get("hyphen-right") or []
        if isinstance(hy_right, list):
            for term in hy_right:
                if isinstance(term, str) and term:
                    HYPHEN_RIGHT_TERMS.add(term)
        prefixes = data.get("prefixes") or data.get("compound_prefixes") or []
        if isinstance(prefixes, list):
            for term in prefixes:
                if isinstance(term, str) and term:
                    COMPOUND_PREFIXES.add(term)

    for term in args.add_hyphen_left or []:
        t = term.strip()
        if t:
            HYPHEN_LEFT_TERMS.add(t)
    for term in args.add_hyphen_right or []:
        t = term.strip()
        if t:
            HYPHEN_RIGHT_TERMS.add(t)


def main(argv: Sequence[str]) -> int:
    args = parse_args(argv)
    load_user_rules(args)
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
