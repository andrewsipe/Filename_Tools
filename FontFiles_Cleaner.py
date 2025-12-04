#!/usr/bin/env python3
"""
Unified font filename cleaner with intelligent pattern matching and auto-adaptive processing.

Intelligently cleans font filenames using pattern matching for heavy lifting and minimal
dictionaries for edge cases. Auto-detects file quality and adapts processing intensity.

Usage:
    python FontFiles_Cleaner.py /path/to/fonts -r -n
    python FontFiles_Cleaner.py /path/to/fonts -r --workflow heavy
    python FontFiles_Cleaner.py /path/to/fonts -r --preview
"""

import argparse
import re
import sys
import unicodedata
from pathlib import Path
from typing import List, Set, Tuple, Dict, Optional

# Import our dictionaries
try:
    from FontCore.core_font_style_dictionaries import (
        COMPOUND_NORMALIZATIONS,
        STYLE_WORDS,
        ALL_WIDTH_TERMS,
        ALL_OPTICAL_TERMS,
        COMPOUND_PREFIXES,
        SLOPE_BASES,
        WIDTH_BASES,
        MODIFIERS,
    )
except ImportError:
    print("ERROR: Could not import core_font_style_dictionaries")
    print("Make sure the core module is in your Python path")
    sys.exit(1)

# Import console styles
try:
    from FontCore.core_console_styles import StatusIndicator

    _HAS_CONSOLE_STYLES = True
except ImportError:
    _HAS_CONSOLE_STYLES = False

    # Fallback StatusIndicator for when core_console_styles not available
    class StatusIndicator:
        def __init__(self, status: str, dry_run: bool = False):
            self.status = status.upper()
            self.dry_run = dry_run
            self.parts = []

        def add_message(self, msg: str, **kwargs) -> "StatusIndicator":
            self.parts.append(msg)
            return self

        def add_file(self, file: str, **kwargs) -> "StatusIndicator":
            self.parts.append(file)
            return self

        def with_summary_block(self, **kwargs) -> "StatusIndicator":
            summary_parts = [f"{k}: {v}" for k, v in kwargs.items()]
            self.parts.append(" | ".join(summary_parts))
            return self

        def build(self) -> str:
            prefix = "DRY-RUN " if self.dry_run else ""
            return f"{prefix}{self.status}: {' '.join(self.parts)}"

        def emit(self) -> None:
            print(self.build())


# ================================================================================================
# SECTION 1: PATTERN MATCHING FUNCTIONS
# ================================================================================================


def is_width_term(token: str) -> bool:
    """
    Check if token is a width term using pattern matching.

    Handles:
    - Base terms (Condensed, Wide, etc.)
    - Modifier + base (SemiCondensed, ExtraWide, etc.)
    - X variations (XCondensed, XXCondensed, etc.)
    """
    if not token:
        return False

    # Check against all generated width variations
    return token in ALL_WIDTH_TERMS


def is_optical_term(token: str) -> bool:
    """Check if token is an optical size term."""
    if not token:
        return False
    return token in ALL_OPTICAL_TERMS


def find_truncation_match(abbrev: str, context: str = "") -> Optional[str]:
    """
    Find full word that abbrev truncates, return best match.

    Uses pattern matching against STYLE_WORDS to find candidates.
    Returns shortest match (most conservative) to avoid over-expansion.
    """
    if not abbrev or not abbrev[0].isalpha():
        return None

    # Skip if already a complete word
    if abbrev in STYLE_WORDS:
        return None

    # Find candidates that start with this abbreviation
    candidates = [
        word
        for word in STYLE_WORDS
        if word.lower().startswith(abbrev.lower()) and len(abbrev) < len(word)
    ]

    if not candidates:
        return None

    # Get shortest match (most conservative)
    best_match = min(candidates, key=len)

    # Safety check: avoid double-expansion (e.g., "Medium" → "Mediumium")
    remainder = best_match[len(abbrev) :].lower()
    if remainder and remainder in context.lower():
        return None

    return best_match


def is_modifier_compound(token: str) -> bool:
    """Check if token is a modifier compound (SemiBold, ExtraLight, etc.)."""
    if not token or len(token) < 6:  # Minimum: "Semi" + "Bold"
        return False

    for mod in MODIFIERS:
        if token.startswith(mod) and len(token) > len(mod):
            next_char = token[len(mod)]
            if next_char.isupper():
                return True
    return False


def normalize_modifier_compound(token: str) -> Optional[str]:
    """
    Normalize modifier compound casing.

    SemiBold → Semibold (lowercase weight)
    SemiCondensed → SemiCondensed (keep width capitalized)
    """
    for mod in sorted(MODIFIERS, key=len, reverse=True):
        if token.startswith(mod):
            remainder = token[len(mod) :]
            if remainder and remainder[0].isupper():
                first_char = remainder[0]

                # Keep capitalized if width/italic indicator (C, E, N, W, I)
                if first_char in {"C", "E", "N", "W", "I"}:
                    return None  # Already correct

                # Lowercase for weights
                return f"{mod}{remainder.lower()}"

    return None


# ================================================================================================
# SECTION 2: OPERATION FUNCTIONS
# ================================================================================================


def _split_stem_and_suffixes(filename: str) -> Tuple[str, str]:
    """Split filename into stem and file extensions."""
    path = Path(filename)
    suffixes = "".join(path.suffixes)
    if not suffixes:
        return filename, ""
    stem = path.name[: -len(suffixes)]
    return stem, suffixes


def _tokenize(text: str) -> List[str]:
    """Split text into alpha and non-alpha segments, preserving delimiters."""
    if not text:
        return []

    parts = []
    buf = []
    is_alpha_prev = text[0].isalpha()

    for ch in text:
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


def expand_abbreviations(filename: str) -> Tuple[str, bool]:
    """
    Expand abbreviated terms using pattern matching.

    Uses find_truncation_match() for smart expansion.
    Avoids double-expansion (Med + ium = Medium, not Mediumium).
    """
    stem, suffixes = _split_stem_and_suffixes(filename)
    tokens = _tokenize(stem)

    new_tokens = []
    changed = False

    for token in tokens:
        if not token or not token[0].isalpha():
            new_tokens.append(token)
            continue

        # Try to find expansion
        expansion = find_truncation_match(token, stem)
        if expansion:
            new_tokens.append(expansion)
            changed = True
        else:
            new_tokens.append(token)

    new_stem = "".join(new_tokens)
    return f"{new_stem}{suffixes}", changed


def normalize_delimiters(filename: str) -> Tuple[str, bool]:
    """
    Remove underscores and spaces, preserve hyphens.
    Preserve existing PascalCase capitalization.

    Only capitalize if:
    - After hyphen AND next char is lowercase
    - After underscore/space removal AND next char is lowercase

    Examples:
        helvetica_neue-bold.ttf      → HelveticaNeue-Bold.ttf
        HelveticaNeue-BoldItalic.ttf → HelveticaNeue-BoldItalic.ttf (no change)
        my_font-semibold.ttf         → MyFont-Semibold.ttf
    """
    stem, suffixes = _split_stem_and_suffixes(filename)

    if not stem:
        return filename, False

    out = []
    capitalize_next = False

    for ch in stem:
        if ch == "-":
            out.append(ch)
            capitalize_next = True  # Will capitalize if next is lowercase
            continue

        if ch in ("_", " "):
            # Remove delimiter, mark to capitalize next
            capitalize_next = True
            continue

        if capitalize_next and ch.isalpha():
            if ch.islower():
                out.append(ch.upper())  # Only capitalize if it's lowercase
            else:
                out.append(ch)  # Preserve existing uppercase
            capitalize_next = False
        else:
            out.append(ch)

    new_stem = "".join(out)
    new_filename = f"{new_stem}{suffixes}"
    return new_filename, new_filename != filename


def normalize_compounds(filename: str, patterns: Dict = None) -> Tuple[str, bool]:
    """
    Fix compound words using dictionary + pattern analysis.

    Conservative approach:
    - Only normalize if in dictionary OR clearly a weight term
    - Preserve unknown terms (like "Very")
    """
    stem, suffixes = _split_stem_and_suffixes(filename)

    if not stem:
        return filename, False

    changes = []
    new_stem = stem

    # Apply dictionary normalizations ONLY
    for src, dst in sorted(
        COMPOUND_NORMALIZATIONS.items(), key=lambda x: len(x[0]), reverse=True
    ):
        if src in new_stem:
            new_stem = new_stem.replace(src, dst)
            changes.append((src, dst))

    # Pattern-based normalization (if patterns provided)
    if patterns:
        tokens = _tokenize(stem)
        for token in tokens:
            if should_normalize_compound(token, patterns):
                normalized = normalize_modifier_compound(token)
                if normalized:
                    new_stem = new_stem.replace(token, normalized)
                    changes.append((token, normalized))

    new_filename = f"{new_stem}{suffixes}"
    return new_filename, new_filename != filename


def normalize_multiple_hyphens(stem: str) -> str:
    """
    Only fix ACTUAL problems with hyphens.

    Problems to fix:
    - Consecutive hyphens (-- → -)
    - Trailing/leading hyphens

    DON'T merge single hyphens (VanillaX-Extralight stays as-is)
    """
    # Collapse consecutive hyphens
    stem = re.sub(r"-{2,}", "-", stem)

    # Remove trailing/leading hyphens
    stem = stem.strip("-")

    # DON'T merge multiple hyphens into family
    # That's an aggressive operation that should be opt-in

    return stem


def fix_hyphenation(filename: str) -> Tuple[str, bool]:
    """
    1. Remove hyphen after compound prefixes (Semi-Bold → SemiBold)
    2. Normalize to single hyphen between family and style
    """
    stem, suffixes = _split_stem_and_suffixes(filename)

    if not stem:
        return filename, False

    changed = False
    new_stem = stem

    # Step 1: Remove hyphen after compound prefixes (Semi-Bold → SemiBold)
    if COMPOUND_PREFIXES:
        alt = "|".join(sorted(COMPOUND_PREFIXES, key=len, reverse=True))
        pattern = re.compile(rf"({alt})-(?=[A-Za-z])")
        before = new_stem
        new_stem = pattern.sub(r"\1", new_stem)
        if new_stem != before:
            changed = True

    # Step 2: Normalize to single hyphen
    before = new_stem
    new_stem = normalize_multiple_hyphens(new_stem)
    if new_stem != before:
        changed = True

    new_filename = f"{new_stem}{suffixes}"
    return new_filename, changed


def _add_left_hyphen(text: str, term: str) -> str:
    """Add hyphen before term if missing and preceded by alpha."""
    i = 0
    while True:
        idx = text.find(term, i)
        if idx == -1:
            break

        # Check if hyphen needed
        if idx > 0 and text[idx - 1].isalpha():
            text = text[:idx] + "-" + text[idx:]
            i = idx + len(term) + 1
        else:
            i = idx + len(term)

    return text


def _add_right_hyphen(text: str, term: str) -> str:
    """Add hyphen after term if missing and followed by alpha."""
    i = 0
    while True:
        idx = text.find(term, i)
        if idx == -1:
            break

        end = idx + len(term)
        if end < len(text) and text[end].isalpha():
            text = text[:end] + "-" + text[end:]
            i = end + 1
        else:
            i = end

    return text


def reorder_widths(filename: str) -> Tuple[str, bool]:
    """
    Move width terms to start of style part (after hyphen).

    Helvetica-BoldCondensed.ttf → Helvetica-CondensedBold.ttf
    """
    stem, suffixes = _split_stem_and_suffixes(filename)

    if "-" not in stem:
        return filename, False

    family, style = stem.rsplit("-", 1)

    # Handle empty style part
    if not style or not style.strip():
        return filename, False

    # Extract width terms from style part
    width_matches = _extract_width_terms(style)

    if not width_matches:
        return filename, False

    # Extract width terms in order
    widths = [term for _, _, term in width_matches]

    # Build remaining text
    remaining_parts = []
    last_pos = 0

    for start, end, _ in width_matches:
        if start > last_pos:
            remaining_parts.append(style[last_pos:start])
        last_pos = end

    if last_pos < len(style):
        remaining_parts.append(style[last_pos:])

    remaining = "".join(remaining_parts)

    # Rebuild: widths first, then remaining
    new_style = "".join(widths) + remaining
    new_stem = f"{family}-{new_style}"
    new_filename = f"{new_stem}{suffixes}"

    return new_filename, new_filename != filename


def _extract_width_terms(text: str) -> List[Tuple[int, int, str]]:
    """Extract all width terms from text, returning (start, end, term) tuples."""
    matches = []
    claimed_ranges = []

    # Sort by length (longest first) for proper matching
    sorted_terms = sorted(ALL_WIDTH_TERMS, key=len, reverse=True)

    for term in sorted_terms:
        pos = 0
        while (idx := text.find(term, pos)) != -1:
            end = idx + len(term)

            # Check for overlap
            overlaps = any(
                not (end <= start or idx >= claimed_end)
                for start, claimed_end in claimed_ranges
            )

            if not overlaps:
                matches.append((idx, end, term))
                claimed_ranges.append((idx, end))

            pos = idx + 1

    return sorted(matches, key=lambda x: x[0])


def reorder_optical_sizes(filename: str) -> Tuple[str, bool]:
    """
    Move optical size terms from style part to end of family part.

    Helvetica-BoldDisplay.ttf → HelveticaDisplay-Bold.ttf
    """
    stem, suffixes = _split_stem_and_suffixes(filename)

    if "-" not in stem:
        return filename, False

    family, style = stem.rsplit("-", 1)

    # Handle empty style part
    if not style or not style.strip():
        return filename, False

    # Extract optical size terms from style
    optical_matches = _extract_optical_terms(style)

    if not optical_matches:
        return filename, False

    # Extract optical terms in order
    opticals = [term for _, _, term in optical_matches]

    # Build remaining style
    remaining_parts = []
    last_pos = 0

    for start, end, _ in optical_matches:
        if start > last_pos:
            remaining_parts.append(style[last_pos:start])
        last_pos = end

    if last_pos < len(style):
        remaining_parts.append(style[last_pos:])

    remaining_style = "".join(remaining_parts)

    # Rebuild: family + optical, then style (if any)
    new_family = family + "".join(opticals)

    if remaining_style:
        new_stem = f"{new_family}-{remaining_style}"
    else:
        new_stem = new_family

    new_filename = f"{new_stem}{suffixes}"
    return new_filename, new_filename != filename


def _extract_optical_terms(text: str) -> List[Tuple[int, int, str]]:
    """Extract all optical size terms from text, returning (start, end, term) tuples."""
    matches = []
    claimed_ranges = []

    # Sort by length (longest first) for proper matching
    sorted_terms = sorted(ALL_OPTICAL_TERMS, key=len, reverse=True)

    for term in sorted_terms:
        pos = 0
        while (idx := text.find(term, pos)) != -1:
            end = idx + len(term)

            # Check for overlap
            overlaps = any(
                not (end <= start or idx >= claimed_end)
                for start, claimed_end in claimed_ranges
            )

            if not overlaps:
                matches.append((idx, end, term))
                claimed_ranges.append((idx, end))

            pos = idx + 1

    return sorted(matches, key=lambda x: x[0])


def insert_regular_weight(filename: str) -> Tuple[str, bool]:
    """
    Add Regular after width-only specifications.

    Helvetica-Condensed.ttf → Helvetica-CondensedRegular.ttf
    Helvetica-CondensedItalic.ttf → Helvetica-CondensedRegularItalic.ttf
    """
    stem, suffixes = _split_stem_and_suffixes(filename)

    if "-" not in stem:
        return filename, False

    family, style = stem.rsplit("-", 1)

    # Check if style starts with width term
    matched_width = None
    for width_variation in sorted(ALL_WIDTH_TERMS, key=len, reverse=True):
        if style.startswith(width_variation):
            matched_width = width_variation
            break

    if not matched_width:
        return filename, False

    after_width = style[len(matched_width) :]

    # Case 1: Width is entire style (Condensed)
    if not after_width:
        new_style = f"{matched_width}Regular"
        new_stem = f"{family}-{new_style}"
        new_filename = f"{new_stem}{suffixes}"
        return new_filename, True

    # Case 2: Width followed by slope (CondensedItalic)
    matched_slope = None
    for slope in SLOPE_BASES:
        if after_width.startswith(slope):
            matched_slope = slope
            break

    if matched_slope:
        remaining = after_width[len(matched_slope) :]
        new_style = f"{matched_width}Regular{matched_slope}{remaining}"
        new_stem = f"{family}-{new_style}"
        new_filename = f"{new_stem}{suffixes}"
        return new_filename, True

    # Case 3: Width followed by something else (likely weight)
    # Don't insert Regular
    return filename, False


def deduplicate_words(filename: str) -> Tuple[str, bool]:
    """
    Remove repeated words while preserving delimiters.

    Neptun-Süd-Wide-Süd-Wide-SemiBold.ttf → Neptun-Süd-Wide-SemiBold.ttf
    """
    stem, suffixes = _split_stem_and_suffixes(filename)

    if not stem:
        return filename, False

    new_stem, removed = _deduplicate_words(stem)
    new_filename = f"{new_stem}{suffixes}"

    return new_filename, new_filename != filename


def _deduplicate_words(text: str) -> Tuple[str, List[str]]:
    """Remove duplicate alphabetic segments."""
    parts = _tokenize(text)

    seen = set()
    removed = []
    out_parts = []

    for part in parts:
        if not _is_alpha_part(part):
            out_parts.append(part)
            continue

        key = _normalize_for_compare(part)

        if key in seen:
            # Remove preceding delimiter if present
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

    # Collapse multiple hyphens
    new_text = re.sub(r"-{2,}", "-", new_text)

    return new_text, removed


def _is_alpha_part(text: str) -> bool:
    """Check if text segment is a word."""
    return bool(text) and any(ch.isalpha() for ch in text)


def _normalize_for_compare(token: str) -> str:
    """Normalize for case-insensitive comparison."""
    try:
        return unicodedata.normalize("NFC", token).casefold()
    except Exception:
        return token.lower()


# ================================================================================================
# SECTION 3: PATTERN ANALYSIS
# ================================================================================================


def analyze_directory_patterns(files: List[Path]) -> Dict:
    """
    Analyze all files to discover patterns and context.

    Returns insights like:
    - Common family names
    - Common style terms (including uncommon ones like "Very")
    - Hyphenation patterns
    - Capitalization consistency
    """
    patterns = {
        "families": {},  # Track family names
        "styles": {},  # Track style terms
        "modifiers": {},  # Track modifier usage (Semi, Extra, Ultra + X)
        "hyphens": {},  # Track hyphenation patterns
    }

    for file in files:
        stem = file.stem

        # Analyze family-style split
        if "-" in stem:
            parts = stem.split("-")
            family = parts[0]
            style = "-".join(parts[1:]) if len(parts) > 1 else ""

            patterns["families"][family] = patterns["families"].get(family, 0) + 1

            # Track style terms
            tokens = _tokenize(style)
            for token in tokens:
                if token.isalpha():
                    patterns["styles"][token] = patterns["styles"].get(token, 0) + 1

        # Track modifier compounds
        for mod in MODIFIERS:
            if mod in stem:
                idx = stem.find(mod)
                if idx > 0 and stem[idx - 1].isupper():
                    # Found modifier compound
                    remainder = stem[idx + len(mod) :]
                    if remainder and remainder[0].isupper():
                        compound = mod + remainder.split("-")[0].split("_")[0]
                        patterns["modifiers"][compound] = (
                            patterns["modifiers"].get(compound, 0) + 1
                        )

    return patterns


def should_normalize_compound(token: str, patterns: Dict) -> bool:
    """
    Decide if a compound should be normalized based on patterns.

    Rules:
    1. If it's in COMPOUND_NORMALIZATIONS dictionary → normalize
    2. If remainder is a known width/slope → keep capitalized
    3. If remainder is a known weight → lowercase
    4. If remainder appears frequently in patterns → keep as-is (conservative)
    5. Otherwise → keep as-is (conservative)
    """
    # Check dictionary first
    if token in COMPOUND_NORMALIZATIONS:
        return True

    # Check if it's a modifier compound
    for mod in MODIFIERS:
        if token.startswith(mod):
            remainder = token[len(mod) :]
            remainder_title = remainder.capitalize()

            # Known width/slope → don't normalize
            if remainder_title in WIDTH_BASES or remainder_title in SLOPE_BASES:
                return False

            # Known weight in STYLE_WORDS → normalize
            if remainder_title in STYLE_WORDS:
                # But check if it appears in patterns as-is
                if token in patterns.get("modifiers", {}):
                    # It's used consistently in this directory → keep as-is
                    return False
                return True

            # Unknown term → keep as-is
            return False

    return False


# ================================================================================================
# SECTION 4: VALIDATION FUNCTIONS
# ================================================================================================


def validate_filename(filename: str) -> bool:
    """Validate filename before processing."""
    if not filename or not filename.strip():
        return False

    path = Path(filename)
    if not path.stem:  # Only extension like ".ttf"
        return False

    if path.stem.replace("-", "").replace("_", "").isdigit():  # Only numbers
        return False

    return True


def validate_for_parser(filename: str) -> Tuple[bool, str]:
    """
    Validate filename meets parser requirements.

    Requirements:
    - PascalCase (no spaces or underscores)
    - 0 or 1 hyphen only
    - Each part starts with uppercase

    Returns: (is_valid, error_message)
    """
    stem, _ = _split_stem_and_suffixes(filename)

    if not stem:
        return False, "Empty stem"

    if " " in stem or "_" in stem:
        return False, "Contains spaces/underscores (must be PascalCase)"

    # Parser allows 0 or 1 hyphen (not "must have exactly 1")
    # So we just check it's not excessive
    if stem.count("-") > 1:
        return False, "Multiple hyphens"

    # Each part should start with uppercase OR digit
    for part in stem.split("-"):
        if part and not (part[0].isupper() or part[0].isdigit()):
            return False, f"Part '{part}' doesn't start with uppercase or digit"

    return True, ""


def safe_operation(op_func):
    """Wrapper to catch and log operation failures gracefully."""

    def wrapper(filename: str, patterns: Dict = None) -> Tuple[str, bool]:
        try:
            # Check if function accepts patterns parameter
            import inspect

            sig = inspect.signature(op_func)
            if "patterns" in sig.parameters:
                return op_func(filename, patterns)
            else:
                return op_func(filename)
        except Exception as e:
            print(f"WARNING: Operation {op_func.__name__} failed for {filename}: {e}")
            return filename, False

    return wrapper


# Apply safety wrappers to all operations
def _apply_safety_wrappers():
    """Apply safe_operation wrapper to all operation functions."""
    global \
        expand_abbreviations, \
        normalize_delimiters, \
        normalize_compounds, \
        fix_hyphenation
    global \
        reorder_widths, \
        reorder_optical_sizes, \
        insert_regular_weight, \
        deduplicate_words

    expand_abbreviations = safe_operation(expand_abbreviations)
    normalize_delimiters = safe_operation(normalize_delimiters)
    normalize_compounds = safe_operation(normalize_compounds)
    fix_hyphenation = safe_operation(fix_hyphenation)
    reorder_widths = safe_operation(reorder_widths)
    reorder_optical_sizes = safe_operation(reorder_optical_sizes)
    insert_regular_weight = safe_operation(insert_regular_weight)
    deduplicate_words = safe_operation(deduplicate_words)


# Apply wrappers when module loads
_apply_safety_wrappers()


# ================================================================================================
# SECTION 4: ANALYSIS & AUTO-DETECTION
# ================================================================================================


def analyze_filename(filename: str) -> Dict:
    """
    Detect issues in filename and return severity sFontCore.

    Returns:
        {
            'has_abbreviations': bool,
            'has_lowercase': bool,
            'has_misplaced_widths': bool,
            'has_misplaced_optical': bool,
            'has_duplicates': bool,
            'has_bad_hyphens': bool,
            'severity': float,  # 0.0 (clean) to 1.0 (messy)
        }
    """
    stem = Path(filename).stem

    # Check for abbreviations (short uppercase tokens that could be truncations)
    has_abbreviations = any(
        len(token) <= 3 and token.isupper() and token.isalpha()
        for token in re.findall(r"[A-Z][a-z]*", stem)
    )

    # Check for lowercase after delimiters
    has_lowercase = bool(re.search(r"[-_][a-z]", stem))

    # Check for misplaced width terms (in style part after hyphen)
    has_misplaced_widths = False
    if "-" in stem:
        family, style = stem.rsplit("-", 1)
        width_matches = _extract_width_terms(style)
        has_misplaced_widths = len(width_matches) > 0

    # Check for misplaced optical terms (in style part after hyphen)
    has_misplaced_optical = False
    if "-" in stem:
        family, style = stem.rsplit("-", 1)
        optical_matches = _extract_optical_terms(style)
        has_misplaced_optical = len(optical_matches) > 0

    # Check for duplicate words
    has_duplicates = _has_duplicate_words(stem)

    # Check for bad hyphenation
    has_bad_hyphens = "--" in stem or bool(
        re.search(r"(Semi|Extra|Ultra|Demi|Super)-[A-Z]", stem)
    )

    # Calculate severity (count issues / 6)
    issue_count = sum(
        [
            has_abbreviations,
            has_lowercase,
            has_misplaced_widths,
            has_misplaced_optical,
            has_duplicates,
            has_bad_hyphens,
        ]
    )

    severity = issue_count / 6.0  # 0.0 to 1.0

    return {
        "has_abbreviations": has_abbreviations,
        "has_lowercase": has_lowercase,
        "has_misplaced_widths": has_misplaced_widths,
        "has_misplaced_optical": has_misplaced_optical,
        "has_duplicates": has_duplicates,
        "has_bad_hyphens": has_bad_hyphens,
        "severity": severity,
    }


def _has_duplicate_words(stem: str) -> bool:
    """Check for duplicate words in stem."""
    tokens = _tokenize(stem)
    seen = set()

    for token in tokens:
        if _is_alpha_part(token):
            key = _normalize_for_compare(token)
            if key in seen:
                return True
            seen.add(key)

    return False


def determine_workflow(files: List[Path]) -> str:
    """
    Analyze files and recommend workflow.

    - If files have major issues (lowercase, underscores) → moderate
    - If files are mostly clean → safe
    - Never auto-select aggressive
    """
    if not files:
        return "moderate"

    # Analyze sample of files (first 100 for performance)
    sample = files[:100]

    severities = []
    for file in sample:
        analysis = analyze_filename(file.name)
        severities.append(analysis["severity"])

    avg_severity = sum(severities) / len(severities)

    # Determine workflow (conservative approach)
    if avg_severity > 0.5:
        workflow = "moderate"  # Files need cleanup
        desc = "Files need cleanup (lowercase, underscores, etc.)"
    else:
        workflow = "safe"  # Files are mostly clean
        desc = "Files are mostly clean (minor fixes only)"

    # Show user what was detected
    StatusIndicator("info").add_message(
        f"Auto-detected: {workflow.upper()} workflow"
    ).emit()
    StatusIndicator("info").add_message(f"Analysis: {desc}").emit()
    StatusIndicator("info").add_message(
        f"Average severity: {avg_severity:.2f} (0=clean, 1=messy)"
    ).emit()
    StatusIndicator("info").add_message(
        f"Files analyzed: {len(sample)} of {len(files)} total"
    ).emit()
    print()  # Blank line for spacing

    return workflow


def show_workflow_details(workflow: str) -> None:
    """Show detailed information about the selected workflow and its operations."""
    workflow_operations = {
        "safe": [
            ("normalize_delimiters", "Remove _ and spaces, preserve PascalCase"),
            ("deduplicate", "Remove duplicate words"),
            ("fix_hyphenation", "Fix consecutive/trailing hyphens only"),
        ],
        "moderate": [
            ("normalize_delimiters", "Remove _ and spaces, preserve PascalCase"),
            ("normalize_compounds", "Normalize using dictionary (conservative)"),
            ("deduplicate", "Remove duplicate words"),
            ("fix_hyphenation", "Fix consecutive/trailing hyphens only"),
        ],
        "aggressive": [
            ("expand_abbreviations", "Expand abbreviations"),
            ("normalize_delimiters", "Remove _ and spaces, preserve PascalCase"),
            ("normalize_compounds", "Normalize using dictionary + patterns"),
            ("fix_hyphenation", "Merge multiple hyphens into family"),
            ("reorder_widths", "Move width terms to start"),
            ("reorder_optical", "Move optical sizes to family"),
            ("insert_regular", "Add Regular after width-only"),
            ("deduplicate", "Remove duplicate words"),
        ],
        # Legacy workflows for backward compatibility
        "heavy": [
            (
                "expand_abbreviations",
                "Expand abbreviated terms (Bd → Bold, Cond → Condensed)",
            ),
            ("normalize_delimiters", "Remove _ and spaces, preserve PascalCase"),
            ("normalize_compounds", "Fix compound word casing (SemiBold → Semibold)"),
            ("fix_hyphenation", "Single hyphen between family-style"),
            ("reorder_widths", "Move width terms to start of style part"),
            ("reorder_optical", "Move optical size terms to family part"),
            ("insert_regular", "Add Regular weight after width-only specifications"),
            ("deduplicate", "Remove duplicate words in filenames"),
        ],
        "balanced": [
            (
                "expand_abbreviations",
                "Expand abbreviated terms (Bd → Bold, Cond → Condensed)",
            ),
            ("normalize_delimiters", "Remove _ and spaces, preserve PascalCase"),
            ("normalize_compounds", "Fix compound word casing (SemiBold → Semibold)"),
            ("fix_hyphenation", "Single hyphen between family-style"),
            ("reorder_widths", "Move width terms to start of style part"),
            ("deduplicate", "Remove duplicate words in filenames"),
        ],
        "light": [
            ("normalize_delimiters", "Remove _ and spaces, preserve PascalCase"),
            ("normalize_compounds", "Fix compound word casing (SemiBold → Semibold)"),
            ("fix_hyphenation", "Single hyphen between family-style"),
            ("deduplicate", "Remove duplicate words in filenames"),
        ],
    }

    operations = workflow_operations.get(workflow, workflow_operations["balanced"])

    StatusIndicator("info").add_message(f"Workflow: {workflow.upper()}").emit()
    StatusIndicator("info").add_message(
        f"Operations to apply ({len(operations)}):"
    ).emit()

    for i, (op_name, description) in enumerate(operations, 1):
        print(f"  {i}. {op_name}: {description}")

    print()


def process_files_pipeline(
    files: List[Path],
    workflow: str,
    disabled_ops: Set[str],
    dry_run: bool = False,
    verbose: bool = False,
) -> Tuple[int, int, int]:
    """
    Process files through complete workflow pipeline with step-by-step logging.

    Returns: (total_processed, total_changed, total_errors)
    """
    total_processed, total_changed, total_errors = 0, 0, 0

    # Analyze patterns for compound normalization (if moderate/aggressive workflow)
    patterns = None
    if workflow in ["moderate", "aggressive"]:
        patterns = analyze_directory_patterns(files)

    for file_path in files:
        total_processed += 1
        original_name = file_path.name

        # Validate filename before processing
        if not validate_filename(original_name):
            if verbose:
                StatusIndicator("skipped").add_file(original_name).with_explanation(
                    "Invalid filename format"
                ).emit()
            continue

        # Execute the complete pipeline
        final_name, did_change, step_log = process_filename(
            original_name, workflow, disabled_ops, patterns
        )

        # Skip unchanged files unless verbose
        if not did_change:
            if verbose:
                StatusIndicator("unchanged").add_file(original_name).emit()
            continue

        # File was changed: log it and/or rename it
        total_changed += 1

        # Create status indicator with step log
        indicator = (
            StatusIndicator("updated", dry_run=dry_run)
            .add_file(original_name)
            .add_message(f"→ {final_name}")
            .add_step_log(step_log)
        )

        if dry_run:
            indicator.emit()
            continue

        # Validate parser compatibility before renaming
        is_valid, error = validate_for_parser(final_name)
        if not is_valid:
            total_errors += 1
            StatusIndicator("error").add_file(original_name).with_explanation(
                f"Parser validation failed: {error}"
            ).emit()
            continue

        # Perform the single, final rename
        try:
            handle_rename(file_path, final_name, "unique")
            indicator.emit()
        except Exception as e:
            total_errors += 1
            StatusIndicator("error").add_file(original_name).with_explanation(
                str(e)
            ).emit()

    return total_processed, total_changed, total_errors


# ================================================================================================
# SECTION 4: WORKFLOW ENGINE
# ================================================================================================


def process_filename(
    filename: str, workflow: str, disabled_ops: Set[str] = None, patterns: Dict = None
) -> Tuple[str, bool, List[Tuple[str, str, str]]]:
    """
    Process filename through appropriate operations with detailed step logging.

    Returns: (final_name, did_change, step_log)
    step_log: List of (operation_name, name_before, name_after) for changes.
    """
    disabled_ops = disabled_ops or set()

    # Define operation sequences for each workflow (conservative approach)
    workflows = {
        "safe": [
            ("normalize_delimiters", normalize_delimiters),
            ("deduplicate", deduplicate_words),
            ("fix_hyphenation", fix_hyphenation),
        ],
        "moderate": [
            ("normalize_delimiters", normalize_delimiters),
            ("normalize_compounds", normalize_compounds),
            ("deduplicate", deduplicate_words),
            ("fix_hyphenation", fix_hyphenation),
        ],
        "aggressive": [
            ("expand_abbreviations", expand_abbreviations),
            ("normalize_delimiters", normalize_delimiters),
            ("normalize_compounds", normalize_compounds),
            ("fix_hyphenation", fix_hyphenation),
            ("reorder_widths", reorder_widths),
            ("reorder_optical", reorder_optical_sizes),
            ("insert_regular", insert_regular_weight),
            ("deduplicate", deduplicate_words),
        ],
        # Legacy workflows for backward compatibility
        "heavy": [
            ("expand_abbreviations", expand_abbreviations),
            ("normalize_delimiters", normalize_delimiters),
            ("normalize_compounds", normalize_compounds),
            ("fix_hyphenation", fix_hyphenation),
            ("reorder_widths", reorder_widths),
            ("reorder_optical", reorder_optical_sizes),
            ("insert_regular", insert_regular_weight),
            ("deduplicate", deduplicate_words),
        ],
        "balanced": [
            ("expand_abbreviations", expand_abbreviations),
            ("normalize_delimiters", normalize_delimiters),
            ("normalize_compounds", normalize_compounds),
            ("fix_hyphenation", fix_hyphenation),
            ("reorder_widths", reorder_widths),
            ("deduplicate", deduplicate_words),
        ],
        "light": [
            ("normalize_delimiters", normalize_delimiters),
            ("normalize_compounds", normalize_compounds),
            ("fix_hyphenation", fix_hyphenation),
            ("deduplicate", deduplicate_words),
        ],
    }

    operations = workflows.get(workflow, workflows["balanced"])

    # Process through operations with step logging
    current = filename
    step_log = []
    did_change = False

    for op_name, op_func in operations:
        if op_name in disabled_ops:
            continue

        before = current

        # Pass patterns to normalize_compounds if available
        if op_name == "normalize_compounds" and patterns:
            new_name, changed = op_func(current, patterns)
        else:
            new_name, changed = op_func(current)

        # Double-check that it actually changed (compare stems only)
        before_stem, _ = _split_stem_and_suffixes(before)
        new_stem, _ = _split_stem_and_suffixes(new_name)

        if changed and new_stem != before_stem:
            step_log.append((op_name, before, new_name))
            current = new_name
            did_change = True
        elif changed:
            # Operation reported change but stems are identical - this indicates a bug
            print(
                f"WARNING: Operation {op_name} reported change but stems match: {before_stem}"
            )

    return current, did_change, step_log


# ================================================================================================
# SECTION 5: CLI & MAIN
# ================================================================================================


def collect_files(paths: List[str], recursive: bool = False) -> List[Path]:
    """Collect all font files from given paths."""
    files = []

    for path_str in paths:
        path = Path(path_str)
        if not path.exists():
            StatusIndicator("warning").add_message(f"Path not found: {path_str}").emit()
            continue

        if path.is_file():
            if path.suffix.lower() in {".ttf", ".otf", ".woff", ".woff2"}:
                files.append(path)
        elif path.is_dir():
            pattern = "**/*" if recursive else "*"
            for p in path.glob(pattern):
                if p.is_file() and p.suffix.lower() in {
                    ".ttf",
                    ".otf",
                    ".woff",
                    ".woff2",
                }:
                    files.append(p)

    return files


def preview_changes(
    files: List[Path], workflow: str, disabled_ops: Set[str], limit: int = 10
) -> None:
    """Show preview of what would change."""
    StatusIndicator("info").add_message(
        f"Preview of changes (first {limit} files with changes):"
    ).emit()
    print()

    shown = 0
    for file in files:
        if shown >= limit:
            break

        new_name, changed, step_log = process_filename(
            file.name, workflow, disabled_ops
        )

        if changed:
            print(f"  {file.name}")
            print(f"  → {new_name}")
            if step_log:
                ops = [op_name for op_name, _, _ in step_log]
                print(f"     [{', '.join(ops)}]")
            print()
            shown += 1

    if shown == 0:
        print("  No changes needed!")
        print()


def ensure_unique_destination(path: Path) -> Path:
    """Generate unique filename by appending (n)."""
    if not path.exists():
        return path

    counter = 1
    stem = path.stem
    suffixes = "".join(path.suffixes)

    candidate = path
    while candidate.exists():
        candidate = path.with_name(f"{stem} ({counter}){suffixes}")
        counter += 1

    return candidate


def handle_rename(
    file_path: Path, new_name: str, conflict_strategy: str = "unique"
) -> Path:
    """Handle file rename with conflict resolution."""
    destination = file_path.with_name(new_name)

    if destination.exists():
        if conflict_strategy == "skip":
            raise FileExistsError(f"Target exists: {new_name}")
        elif conflict_strategy == "unique":
            destination = ensure_unique_destination(destination)
        # overwrite: proceed

    file_path.rename(destination)
    return destination


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Intelligent font filename cleaner with auto-detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-detect and preview
  %(prog)s /fonts -r -n
  
  # Auto-detect and apply
  %(prog)s /fonts -r
  
  # Force specific workflow
  %(prog)s /fonts -r --workflow heavy
  
  # Disable specific operations
  %(prog)s /fonts -r --no-reorder-widths
        """,
    )

    # Positional arguments
    parser.add_argument("paths", nargs="+", help="Files/directories to process")

    # General options
    parser.add_argument(
        "-r", "--recursive", action="store_true", help="Recurse into directories"
    )
    parser.add_argument(
        "-n",
        "--dry-run",
        action="store_true",
        help="Show what would change without renaming",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Show unchanged files and additional info",
    )
    parser.add_argument(
        "--workflow",
        choices=[
            "auto",
            "safe",
            "moderate",
            "aggressive",
            "heavy",
            "balanced",
            "light",
        ],
        default="auto",
        help="Processing intensity (default: auto)",
    )
    parser.add_argument(
        "--preview", action="store_true", help="Show preview of changes before applying"
    )
    parser.add_argument(
        "--conflict",
        choices=["skip", "unique", "overwrite"],
        default="unique",
        help="On name conflict: skip, unique (n), or overwrite (default: unique)",
    )

    # Operation toggles
    parser.add_argument(
        "--no-expand-abbrev", action="store_true", dest="no_expand_abbreviations"
    )
    parser.add_argument("--no-capitalize", action="store_true")
    parser.add_argument("--no-normalize-compounds", action="store_true")
    parser.add_argument("--no-fix-hyphens", action="store_true")
    parser.add_argument("--no-reorder-widths", action="store_true")
    parser.add_argument("--no-reorder-optical", action="store_true")
    parser.add_argument("--no-insert-regular", action="store_true")
    parser.add_argument("--no-deduplicate", action="store_true")

    args = parser.parse_args()

    # Collect files
    files = collect_files(args.paths, args.recursive)
    StatusIndicator("info").add_message(f"Found {len(files)} files").emit()

    if not files:
        StatusIndicator("error").add_message("No font files found").emit()
        return 1

    # Determine workflow
    if args.workflow == "auto":
        workflow = determine_workflow(files)
    else:
        workflow = args.workflow
        StatusIndicator("info").add_message(
            f"Using {workflow.upper()} workflow (manual override)"
        ).emit()
        print()

    # Show workflow details
    show_workflow_details(workflow)

    # Build disabled operations set
    disabled = set()
    if args.no_expand_abbreviations:
        disabled.add("expand_abbreviations")
    if args.no_capitalize:
        disabled.add("capitalize")
    if args.no_normalize_compounds:
        disabled.add("normalize_compounds")
    if args.no_fix_hyphens:
        disabled.add("fix_hyphens")
    if args.no_reorder_widths:
        disabled.add("reorder_widths")
    if args.no_reorder_optical:
        disabled.add("reorder_optical")
    if args.no_insert_regular:
        disabled.add("insert_regular")
    if args.no_deduplicate:
        disabled.add("deduplicate")

    # Preview mode
    if args.preview or args.dry_run:
        if args.preview:
            return 0

    # Process files through complete pipeline
    total, changed, errors = process_files_pipeline(
        files, workflow, disabled, dry_run=args.dry_run, verbose=args.verbose
    )

    # Final summary
    print("=" * 80)
    StatusIndicator("success").add_message("Processing Complete!").emit()
    print("=" * 80)
    StatusIndicator("info").add_message(f"Total files processed: {total}").emit()
    StatusIndicator("info").add_message(f"Files updated: {changed}").emit()
    StatusIndicator("info").add_message(f"Files unchanged: {total - changed}").emit()
    StatusIndicator("info").add_message(f"Errors: {errors}").emit()
    print("=" * 80)

    return 0 if errors == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
