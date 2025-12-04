#!/usr/bin/env python3
"""
Expand common font style abbreviations in filenames (stem only), preserving delimiters.

Focus: Abbreviation expansion only.

Pipeline:
- Apply non-truncated rules, then truncated rules (with lookahead safety) to avoid
  double-expansions like Mediumium.

CLI:
- paths... [-r] [-n] [--conflict skip|unique|overwrite] [-v] [--cleanup|--no-cleanup]
- Optional dictionary tuning: --rules-file, --add-rule ABBR=EXPANSION
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# Add project root to path for FontCore imports (works for root and subdirectory scripts)
_project_root = Path(__file__).parent
while not (_project_root / "FontCore").exists() and _project_root.parent != _project_root:
    _project_root = _project_root.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import FontCore.core_console_styles as cs


# --- Rules (copied from Files_ExpandStyleNames.py) ----------------------------------------------

ABBREV_RULES: Dict[str, str] = {
    "Blk": "Black",
    "Bd": "Bold",
    "Bk": "Book",
    "Cd": "Condensed",
    "Cn": "Condensed",
    "Cnd": "Condensed",
    "Cond": "Condensed",
    "DBb": "Demibold",
    "DmBd": "Demibold",
    "DemBd": "Demibold",
    "DmBold": "Demibold",
    "DemBold": "Demibold",
    "Exp": "Expanded",
    "XCn": "ExtraCondensed",
    "XCnd": "ExtraCondensed",
    "XCond": "ExtraCondensed",
    "ExCn": "ExtraCondensed",
    "ExCnd": "ExtraCondensed",
    "ExCond": "ExtraCondensed",
    "ExtraCond": "ExtraCondensed",
    "XBd": "Extrabold",
    "XBold": "Extrabold",
    "ExBd": "Extrabold",
    "ExBold": "Extrabold",
    "XLt": "Extralight",
    "ExLt": "Extralight",
    "XLight": "Extralight",
    "ExLight": "Extralight",
    "It": "Italic",
    "Ita": "Italic",
    "italic": "Italic",
    "Lt": "Light",
    "Md": "Medium",
    "Med": "Medium",
    "Nr": "Narrow",
    "Nar": "Narrow",
    "Obl": "Oblique",
    "Obliq": "Oblique",
    "Rg": "Regular",
    "Reg": "Regular",
    "Rnd": "Round",
    "SmBd": "Semibold",
    "SemBd": "Semibold",
    "SemiBd": "Semibold",
    "SmBold": "Semibold",
    "SemBold": "Semibold",
    "SmLt": "Semilight",
    "SemLt": "Semilight",
    "SemiLt": "Semilight",
    "SmLight": "Semilight",
    "SemLight": "Semilight",
    "SmCd": "SemiCondensed",
    "SmCnd": "SemiCondensed",
    "SmCond": "SemiCondensed",
    "SemiCd": "SemiCondensed",
    "SemiCnd": "SemiCondensed",
    "SemiCond": "SemiCondensed",
    "Ult": "Ultra",
    "UltBd": "Ultrabold",
    "UltBold": "Ultrabold",
    "UltLt": "Ultralight",
    "UltLight": "Ultralight",
    "UltraCond": "UltraCondensed",
    "GX": "Variable",
    "VF": "Variable",
    "Var": "Variable",
    "Wd": "Wide",
    "Wid": "Wide",
    "SC": "Smallcaps",
}


# Partition rules into truncated vs non-truncated (computed at load and after user updates)
NON_TRUNCATED_RULES: Dict[str, str] = {}
TRUNCATED_RULES: Dict[str, str] = {}


def recompute_rule_partitions() -> None:
    global NON_TRUNCATED_RULES, TRUNCATED_RULES
    non_trunc: Dict[str, str] = {}
    trunc: Dict[str, str] = {}
    for abbr, exp in ABBREV_RULES.items():
        try:
            if exp.lower().startswith(abbr.lower()) and len(abbr) < len(exp):
                trunc[abbr] = exp
            else:
                non_trunc[abbr] = exp
        except Exception:
            non_trunc[abbr] = exp
    NON_TRUNCATED_RULES = non_trunc
    TRUNCATED_RULES = trunc


recompute_rule_partitions()


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


def _lookahead_matches_remainder(
    parts: List[str],
    index: int,
    token: str,
    abbrev_low: str,
    expansion_low: str,
) -> bool:
    remainder_needed = expansion_low[len(abbrev_low) :]
    if not remainder_needed:
        return False
    token_low = token.lower()
    in_token_remainder = token_low[len(abbrev_low) :]
    collected = in_token_remainder
    j = index + 1
    while len(collected) < len(remainder_needed) and j < len(parts):
        nxt = parts[j]
        if _is_alpha_part(nxt):
            collected += nxt.lower()
        j += 1
    if len(collected) < len(remainder_needed):
        return False
    return collected[: len(remainder_needed)] == remainder_needed


def safe_expand_abbrev(
    parts: List[str],
    index: int,
    token: str,
    abbrev: str,
    expansion: str,
) -> Optional[str]:
    if not token or not abbrev:
        return None
    token_low = token.lower()
    abbrev_low = abbrev.lower()
    expansion_low = expansion.lower()
    if token_low.startswith(expansion_low):
        return None
    if token_low == abbrev_low:
        if expansion_low.startswith(abbrev_low) and _lookahead_matches_remainder(
            parts, index, token, abbrev_low, expansion_low
        ):
            return None
        return expansion
    if token_low.startswith(abbrev_low):
        remainder_current = token[len(abbrev) :]
        if not remainder_current or remainder_current[0].isupper():
            if expansion_low.startswith(abbrev_low) and _lookahead_matches_remainder(
                parts, index, token, abbrev_low, expansion_low
            ):
                return None
            return f"{expansion}{remainder_current}"
    if (
        expansion_low.startswith(abbrev_low)
        and len(abbrev) < len(expansion)
        and token_low.endswith(abbrev_low)
    ):
        pos = len(token) - len(abbrev)
        if pos >= 1 and token[pos].isupper():
            return f"{token[:pos]}{expansion}"
    if token_low.endswith(abbrev_low):
        pos = len(token) - len(abbrev)
        if pos >= 1 and token[pos].isupper():
            return f"{token[:pos]}{expansion}"
    pos_infix = token.find(abbrev)
    if pos_infix > 0:
        end = pos_infix + len(abbrev)
        left_boundary = token[pos_infix - 1].islower() and token[pos_infix].isupper()
        right_boundary = (end == len(token)) or token[end].isupper()
        if left_boundary and right_boundary:
            return f"{token[:pos_infix]}{expansion}{token[end:]}"
    return None


def apply_abbreviation_rules_to_basename(
    basename: str, rules: Dict[str, str]
) -> Tuple[str, List[Tuple[str, str]]]:
    parts = tokenize_style_segments(basename)
    if not parts:
        return basename, []
    changes: List[Tuple[str, str]] = []
    new_parts: List[str] = []
    for idx, part in enumerate(parts):
        if not _is_alpha_part(part):
            new_parts.append(part)
            continue
        replaced = False
        for abbrev in sorted(rules.keys(), key=len, reverse=True):
            expansion = rules[abbrev]
            candidate = safe_expand_abbrev(parts, idx, part, abbrev, expansion)
            if candidate is not None and candidate != part:
                changes.append((part, candidate))
                new_parts.append(candidate)
                replaced = True
                break
        if not replaced:
            new_parts.append(part)
    new_basename = "".join(new_parts)
    return new_basename, changes


@dataclass
class RenameDecision:
    old_name: str
    new_name: str
    changes: List[Tuple[str, str]]
    destination: Path


def build_new_filename(original_name: str) -> Tuple[str, List[Tuple[str, str]]]:
    stem, suffixes = split_stem_and_suffixes(original_name)
    after_nontrunc, changes1 = apply_abbreviation_rules_to_basename(
        stem, NON_TRUNCATED_RULES
    )
    after_trunc, changes2 = apply_abbreviation_rules_to_basename(
        after_nontrunc, TRUNCATED_RULES
    )
    return f"{after_trunc}{suffixes}", (changes1 + changes2)


def compute_rename(file_path: Path) -> RenameDecision:
    old_name = file_path.name
    new_name, changes = build_new_filename(old_name)
    destination = file_path.with_name(new_name)
    return RenameDecision(
        old_name=old_name, new_name=new_name, changes=changes, destination=destination
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


def perform_rename(
    file_path: Path,
    *,
    dry_run: bool,
    conflict: str,
    verbose: bool,
) -> Tuple[bool, str | None]:
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
            destination = ensure_unique_destination(destination)
        # overwrite: proceed

    if dry_run:
        cs.StatusIndicator("info", dry_run=True).add_message(
            f"DRY-RUN rename in {cs.fmt_file(str(file_path.parent))}"
        ).add_values(old_value=decision.old_name, new_value=destination.name).emit()
        return True, None

    try:
        file_path.rename(destination)
        cs.StatusIndicator("updated").add_message(
            f"in {cs.fmt_file(str(file_path.parent))}"
        ).add_values(old_value=decision.old_name, new_value=destination.name).emit()
        return True, None
    except Exception as exc:  # noqa: BLE001
        return False, f"rename failed for {file_path}: {exc}"


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("Expand common font style abbreviations in filenames (stem only).")
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
        help="JSON file with {abbreviations} overrides/additions",
    )
    parser.add_argument(
        "--add-rule",
        action="append",
        default=[],
        metavar="ABBR=EXPANSION",
        help="Add or override a single abbreviation rule (can repeat)",
    )
    return parser.parse_args(argv)


def load_user_rules(args: argparse.Namespace) -> None:
    for raw in args.add_rule or []:
        if "=" not in raw:
            cs.StatusIndicator("warning").with_explanation(
                f"invalid --add-rule (expected ABBR=EXPANSION): {raw}"
            ).emit()
            continue
        abbr, exp = raw.split("=", 1)
        abbr = abbr.strip()
        exp = exp.strip()
        if not abbr or not exp:
            cs.StatusIndicator("warning").with_explanation(
                f"invalid --add-rule (empty key/value): {raw}"
            ).emit()
            continue
        ABBREV_RULES[abbr] = exp

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
        abbrs = data.get("abbreviations") or data.get("abbr") or {}
        if isinstance(abbrs, dict):
            for k, v in abbrs.items():
                if isinstance(k, str) and isinstance(v, str):
                    ABBREV_RULES[k] = v

    recompute_rule_partitions()


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
