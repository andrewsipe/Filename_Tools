#!/usr/bin/env python3
"""
Simple Filename Corpus Pattern Analyzer
Focuses on finding obvious style terms through n-gram suffix analysis.

Usage:
    python Filename_CorpusPatternAnalyzer_Simple.py <path> [<path2> ...] [--recursive]

Outputs: corpus_analysis_report.md
"""

import sys
import re
from collections import defaultdict, Counter
from datetime import datetime
from pathlib import Path

# Add project root to path for FontCore imports (works for root and subdirectory scripts)
# ruff: noqa: E402
_project_root = Path(__file__).parent
while (
    not (_project_root / "FontCore").exists() and _project_root.parent != _project_root
):
    _project_root = _project_root.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# Import core file collector
from FontCore.core_file_collector import collect_font_files


# ============================================================================
# DICTIONARY - Edit these lists to match your style terms
# ============================================================================

DICTIONARY = {
    "Modifier": [
        "Extra",
        "Ultra",
        "Demi",
        "Semi",
        "X",
    ],
    "Optical_Sizes": [
        "Caption",
        "Display",
        "Text",
        "Poster",
        "Headline",
        "Subhead",
        "Title",
        "Titling",
        "Deck",
        "Micro",
        "Banner",
        "Fine",
        "Large",
        "Small",
        "Big",
        "Tall",
    ],
    "Width": [
        "Compressed",
        "Condensed",
        "Compact",
        "Narrow",
        "Tight",
        "Extended",
        "Expanded",
        "Wide",
        "SemiCondensed",
        "ExtraCondensed",
        "UltraCondensed",
        "SemiCompressed",
        "ExtraCompressed",
        "UltraCompressed",
        "SemiCompact",
        "ExtraCompact",
        "UltraCompact",
        "SemiNarrow",
        "ExtraNarrow",
        "UltraNarrow",
        "SemiExpanded",
        "ExtraExpanded",
        "UltraExpanded",
        "SemiExtended",
        "ExtraExtended",
        "UltraExtended",
        "SemiWide",
        "ExtraWide",
        "UltraWide",
    ],
    "Weight": [
        "Thin",
        "Hairline",
        "Extralight",
        "Ultralight",
        "Light",
        "Semilight",
        "Book",
        "Regular",
        "Normal",
        "Medium",
        "Demibold",
        "Semibold",
        "Bold",
        "Extrabold",
        "Ultrabold",
        "Black",
        "Heavy",
        "Extrablack",
        "Ultrablack",
    ],
    "Slope": [
        "Italic",
        "Oblique",
        "Slanted",
        "Slant",
        "Inclined",
        "Backslanted",
        "Backslant",
        "Reverse",
        "Retalic",
    ],
}

# ============================================================================


def strip_extension(filename):
    """Remove common font file extensions."""
    return re.sub(r"\.(otf|ttf|woff|woff2)$", "", filename, flags=re.IGNORECASE)


def detect_delimiters(text):
    """Detect which delimiters are present in the text."""
    delims = []
    if "-" in text:
        delims.append("-")
    if "_" in text:
        delims.append("_")
    if " " in text:
        delims.append(" ")
    return delims


def extract_ngrams(text, n):
    """Extract all n-grams of length n from text."""
    text = text.lower()
    if len(text) < n:
        return []
    return [text[i : i + n] for i in range(len(text) - n + 1)]


def find_common_prefixes(stems, min_length=4, min_occurrences=2):
    """Find common prefixes (potential family names)."""
    prefix_counts = defaultdict(int)

    for i in range(len(stems)):
        for j in range(i + 1, len(stems)):
            prefix = ""
            min_len = min(len(stems[i]), len(stems[j]))
            for k in range(min_len):
                if stems[i][k] == stems[j][k]:
                    prefix += stems[i][k]
                else:
                    break

            if len(prefix) >= min_length:
                prefix_counts[prefix] += 1

    candidates = [
        (prefix, count)
        for prefix, count in prefix_counts.items()
        if count >= min_occurrences
    ]
    candidates.sort(key=lambda x: (-x[1], -len(x[0])))

    return candidates[:5]


def extract_terms_single_pass(stems, min_files=50, dropoff_threshold=0.4):
    """
    Single pass of n-gram suffix analysis with frequency-based boundary detection.
    Returns discovered terms for this pass.
    """
    # Collect n-grams for lengths 3-8 (covers most style terms)
    all_ngrams = Counter()

    for stem in stems:
        stem_lower = stem.lower()
        for length in range(3, 11):  # 3 to 8 chars
            if len(stem_lower) >= length:
                ngram = stem_lower[-length:]  # Take from end (suffix)
                all_ngrams[ngram] += 1

    # Filter to n-grams that appear frequently
    threshold = max(min_files, len(stems) * 0.05)  # At least 5% or min_files
    candidates = []

    for ngram, count in all_ngrams.items():
        if count < threshold:
            continue

        # Check what % of occurrences are at the end of filenames
        suffix_count = sum(1 for s in stems if s.lower().endswith(ngram))
        suffix_ratio = suffix_count / count

        # Only keep if it's primarily a suffix (80%+)
        if suffix_ratio >= 0.8:
            candidates.append((ngram, count, suffix_ratio))

    if not candidates:
        return []

    # Sort by length (SHORTER first) to process roots before compounds
    candidates.sort(key=lambda x: (len(x[0]), -x[1]))

    # Frequency-based boundary detection
    final_terms = []
    skip_terms = set()  # Terms to skip (they're substrings or extensions)

    for term, count, ratio in candidates:
        if term in skip_terms:
            continue

        # Check if this term is a boundary (stable frequency)
        is_boundary = True

        # Look for longer versions of this term in candidates
        for longer_term, longer_count, _ in candidates:
            if len(longer_term) > len(term) and longer_term.endswith(term):
                # Calculate frequency drop when adding characters
                frequency_drop = (count - longer_count) / count

                # If frequency is stable (drop < threshold), this isn't the boundary
                if frequency_drop < dropoff_threshold:
                    is_boundary = False
                    skip_terms.add(term)  # Skip this, wait for the longer version
                    break
                else:
                    # Sharp drop detected - mark longer term as extension
                    skip_terms.add(longer_term)

        if is_boundary:
            final_terms.append((term, count, ratio))

    # Sort by frequency (most common first)
    final_terms.sort(key=lambda x: -x[1])

    return final_terms


def extract_style_terms(stems, min_files=50, dropoff_threshold=0.4, max_passes=10):
    """
    Multi-pass term extraction with progressive stripping.

    Each pass:
    1. Find the most common suffix terms
    2. Strip them from all stems
    3. Repeat until no new terms found

    This exposes terms that were hidden in the middle (like "condensed").
    """
    all_discovered = []
    current_stems = [s.lower() for s in stems]
    pass_num = 1

    while pass_num <= max_passes:
        # Run single pass
        pass_terms = extract_terms_single_pass(
            current_stems, min_files, dropoff_threshold
        )

        if not pass_terms:
            # No new terms found, we're done
            break

        # Add to discovered list with pass info
        for term, count, ratio in pass_terms:
            all_discovered.append((term, count, ratio, pass_num))

        # Strip discovered terms from stems for next pass
        new_stems = []
        for stem in current_stems:
            remaining = stem
            # Strip all newly discovered terms from the end
            for term, _, _ in pass_terms:
                if remaining.endswith(term):
                    remaining = remaining[: -len(term)]

            if remaining:  # Only keep if something remains
                new_stems.append(remaining)

        if not new_stems or new_stems == current_stems:
            # Nothing left to analyze or no change
            break

        current_stems = new_stems
        pass_num += 1

    # Convert to final format (term, count, ratio) - drop pass_num for compatibility
    final_terms = [(term, count, ratio) for term, count, ratio, _ in all_discovered]

    return final_terms


def fuzzy_match_term(discovered_term, dictionary):
    """
    Attempt to match discovered term against dictionary (case-insensitive).
    Returns: (category, matched_term, confidence) or None

    Matching rules:
    1. Exact match (case-insensitive) = 100% confidence
    2. Dictionary term is substring of discovered = 80% confidence
    3. Discovered term is substring of dictionary term = 70% confidence
    4. Edit distance similarity > 60% = similarity% confidence
    """
    from difflib import SequenceMatcher

    best_match = None
    best_confidence = 0
    best_category = None

    discovered_lower = discovered_term.lower()

    for category, terms in dictionary.items():
        for dict_term in terms:
            dict_lower = dict_term.lower()
            confidence = 0

            # Exact match (case-insensitive)
            if discovered_lower == dict_lower:
                confidence = 100
            # Dictionary term is substring of discovered (e.g., "bold" in "semibold")
            elif dict_lower in discovered_lower:
                confidence = 80
            # Discovered is substring of dictionary (e.g., "nded" in "extended")
            elif discovered_lower in dict_lower:
                confidence = 70
            # Edit distance similarity
            else:
                matcher = SequenceMatcher(None, discovered_lower, dict_lower)
                similarity = matcher.ratio() * 100
                if similarity >= 60:
                    confidence = similarity

            if confidence > best_confidence:
                best_confidence = confidence
                best_match = dict_term  # Return original case from dictionary
                best_category = category

    if best_match and best_confidence >= 60:
        return (best_category, best_match, best_confidence)

    return None


def validate_with_dictionary(discovered_terms, dictionary):
    """
    Cross-reference discovered terms with dictionary.
    Returns categorized results with confidence scores.
    """
    exact_matches = []
    fuzzy_matches = []
    unknown_terms = []

    for term, count, ratio in discovered_terms:
        match_result = fuzzy_match_term(term, dictionary)

        if match_result:
            category, matched_term, confidence = match_result

            if confidence == 100:
                exact_matches.append((term, count, category, matched_term, confidence))
            else:
                fuzzy_matches.append((term, count, category, matched_term, confidence))
        else:
            unknown_terms.append((term, count))

    return {"exact": exact_matches, "fuzzy": fuzzy_matches, "unknown": unknown_terms}


def suggest_split(stem, discovered_terms):
    """
    Suggest how to split a merged filename using discovered terms.
    Returns: (family_part, style_parts) or None
    """
    stem_lower = stem.lower()
    matched_terms = []
    remaining = stem_lower

    # Try to match terms from end to start (greedy)
    for term, _, _ in sorted(discovered_terms, key=lambda x: -len(x[0])):
        if remaining.endswith(term):
            matched_terms.insert(0, term)
            remaining = remaining[: -len(term)]

    if matched_terms and remaining:
        return (remaining, matched_terms)

    return None


def analyze_corpus(filenames):
    """Main analysis function."""
    stems = [strip_extension(f.strip()) for f in filenames if f.strip()]

    if not stems:
        return None

    results = {
        "total_files": len(stems),
        "stems": stems,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    # Basic stats
    delimiter_counts = Counter()
    for stem in stems:
        delims = detect_delimiters(stem)
        if not delims:
            delimiter_counts["none"] += 1
        else:
            for d in delims:
                delimiter_counts[d] += 1

    results["delimiter_counts"] = delimiter_counts
    results["avg_length"] = sum(len(s) for s in stems) / len(stems)

    # Family detection
    family_candidates = find_common_prefixes(stems)
    results["family_candidates"] = family_candidates

    results["family_groups"] = {}
    for prefix, _ in family_candidates:
        files = [s for s in stems if s.startswith(prefix)]
        results["family_groups"][prefix] = files

    # CORE: Extract style terms
    discovered_terms = extract_style_terms(stems, min_files=50)
    results["discovered_terms"] = discovered_terms

    # NEW: Dictionary validation
    validation = validate_with_dictionary(discovered_terms, DICTIONARY)
    results["validation"] = validation

    # Sample splits for demonstration
    sample_splits = {}
    for stem in stems[:20]:  # First 20 files
        split = suggest_split(stem, discovered_terms)
        if split:
            sample_splits[stem] = split

    results["sample_splits"] = sample_splits

    # Calculate coverage
    files_with_terms = set()
    for stem in stems:
        stem_lower = stem.lower()
        for term, _, _ in discovered_terms:
            if term in stem_lower:
                files_with_terms.add(stem)
                break

    results["coverage"] = len(files_with_terms) / len(stems) * 100

    return results


def generate_markdown_report(results):
    """Generate simple, focused markdown report."""

    if not results:
        return "# Error\n\nNo valid filenames to analyze."

    md = []
    md.append("# Simplified Corpus Pattern Analysis\n")
    md.append(f"**Generated:** {results['timestamp']}  ")
    md.append(f"**Total Files:** {results['total_files']}  ")
    md.append(f"**Average Length:** {results['avg_length']:.1f} characters\n")
    md.append("---\n")

    # Section 1: Delimiter Overview
    md.append("## 1. Delimiter Usage\n")
    for delim, count in results["delimiter_counts"].most_common():
        display = "None/Merged" if delim == "none" else f'"{delim}"'
        pct = count / results["total_files"] * 100
        md.append(f"- {display}: {count} files ({pct:.0f}%)")
    md.append("")

    # Section 2: Family Detection
    md.append("## 2. Detected Family Clusters\n")
    if results["family_candidates"]:
        for idx, (prefix, count) in enumerate(results["family_candidates"], 1):
            files = results["family_groups"][prefix]
            pct = len(files) / results["total_files"] * 100
            md.append(f'**{idx}. "{prefix}"** - {len(files)} files ({pct:.0f}%)')
            for f in files[:2]:
                md.append(f"  - {f}")
            if len(files) > 2:
                md.append(f"  - ... ({len(files) - 2} more)")
            md.append("")
    else:
        md.append("*No strong patterns detected.*\n")

    # Section 3: DISCOVERED STYLE TERMS (The Important Part)
    md.append("## 3. Discovered Style Terms\n")
    md.append("**Found by n-gram suffix analysis:**\n")

    if results["discovered_terms"]:
        md.append("| Term | Files | % of Corpus | Suffix Ratio | Length |")
        md.append("|------|-------|-------------|--------------|--------|")

        for term, count, ratio in results["discovered_terms"][:20]:
            pct = count / results["total_files"] * 100
            md.append(
                f"| `{term}` | {count} | {pct:.0f}% | {ratio * 100:.0f}% | {len(term)} chars |"
            )

        md.append(
            f"\n**Total Discovered:** {len(results['discovered_terms'])} unique terms"
        )
        md.append(
            f"**Coverage:** {results['coverage']:.0f}% of files contain at least one discovered term\n"
        )
    else:
        md.append(
            "*No terms met the discovery threshold (50 files or 5% of corpus).*\n"
        )

    # NEW Section 4: Dictionary Validation
    if results.get("validation"):
        md.append("## 4. Dictionary Validation\n")

        validation = results["validation"]

        # Exact matches
        if validation["exact"]:
            md.append("### ✓ Exact Matches (100% confidence)\n")
            md.append("*Discovered terms that exactly match dictionary entries:*\n")
            md.append("| Term | Files | Category | Match |")
            md.append("|------|-------|----------|-------|")
            for term, count, category, matched, conf in validation["exact"]:
                md.append(f"| `{term}` | {count} | {category} | `{matched}` |")
            md.append("")

        # Fuzzy matches
        if validation["fuzzy"]:
            md.append("### ~ Fuzzy Matches (60-99% confidence)\n")
            md.append("*Discovered terms that partially match dictionary entries:*\n")
            md.append("| Term | Files | Category | Best Match | Confidence |")
            md.append("|------|-------|----------|------------|------------|")
            for term, count, category, matched, conf in validation["fuzzy"]:
                md.append(
                    f"| `{term}` | {count} | {category} | `{matched}` | {conf:.0f}% |"
                )
            md.append("")

        # Unknown terms
        if validation["unknown"]:
            md.append("### ? Unknown Terms (not in dictionary)\n")
            md.append(
                "*These terms don't match any dictionary entries - may need manual review:*\n"
            )
            md.append("| Term | Files | % of Corpus |")
            md.append("|------|-------|-------------|")
            for term, count in validation["unknown"]:
                pct = count / results["total_files"] * 100
                md.append(f"| `{term}` | {count} | {pct:.0f}% |")
            md.append("")

        # Summary stats
        total = len(results["discovered_terms"])
        exact_count = len(validation["exact"])
        fuzzy_count = len(validation["fuzzy"])
        unknown_count = len(validation["unknown"])

        md.append("**Validation Summary:**")
        md.append(
            f"- Exact matches: {exact_count}/{total} ({exact_count / total * 100:.0f}%)"
        )
        md.append(
            f"- Fuzzy matches: {fuzzy_count}/{total} ({fuzzy_count / total * 100:.0f}%)"
        )
        md.append(
            f"- Unknown terms: {unknown_count}/{total} ({unknown_count / total * 100:.0f}%)"
        )
        md.append("")

    # Section 5: Sample Splits
    if results["sample_splits"]:
        md.append("\n## 5. Sample Filename Splits\n")
        md.append("*Showing how merged filenames could be tokenized:*\n")

        for stem, (family, style_parts) in list(results["sample_splits"].items())[:10]:
            md.append(f"\n**`{stem}`**")
            md.append(f"- Family: `{family}`")
            md.append(f"- Style: `{' + '.join(style_parts)}`")

    # Section 6: Recommendations
    md.append("\n## 6. Recommendations\n")

    if results["discovered_terms"]:
        high_confidence = [
            t
            for t, c, _ in results["discovered_terms"]
            if c > results["total_files"] * 0.15
        ]

        md.append("### Session Dictionary (High Confidence)\n")
        md.append(
            "*Terms appearing in >15% of files - add these to your working dictionary:*\n"
        )

        if high_confidence:
            for term in high_confidence:
                md.append(f"- `{term}`")
        else:
            md.append("*(None above 15% threshold)*")

        md.append("\n### All Discovered Terms\n")
        md.append("*Complete list for manual review:*\n")
        all_terms = [term for term, _, _ in results["discovered_terms"]]
        md.append(f"`{', '.join(all_terms)}`")
    else:
        md.append("### Next Steps\n")
        md.append("- Lower the discovery threshold (currently 50 files)")
        md.append("- Or manually create a seed dictionary for this corpus")

    md.append("\n---\n*End of Report*")

    return "\n".join(md)


def main():
    if len(sys.argv) < 2:
        print(
            "Usage: python Filename_CorpusPatternAnalyzer_Simple.py <path> [--recursive]"
        )
        sys.exit(1)

    args = sys.argv[1:]
    recursive = "--recursive" in args or "-r" in args
    args = [arg for arg in args if arg not in ["--recursive", "-r"]]

    if not args:
        print("Error: No paths provided.")
        sys.exit(1)

    print(f"Collecting font files from {len(args)} path(s)...")
    if recursive:
        print("  (recursive mode enabled)")

    font_files = collect_font_files(args, recursive=recursive)

    if not font_files:
        print("Error: No font files found.")
        sys.exit(1)

    filenames = [Path(f).name for f in font_files]
    print(f"Found {len(filenames)} files. Analyzing...\n")

    # Analyze
    results = analyze_corpus(filenames)

    if not results:
        print("Error: No valid filenames to analyze.")
        sys.exit(1)

    # Generate report
    report = generate_markdown_report(results)

    # Determine output directory
    output_dir = None
    for arg in args:
        path_obj = Path(arg).expanduser()
        if path_obj.is_dir():
            output_dir = path_obj
            break

    if output_dir is None:
        output_dir = Path(font_files[0]).parent if font_files else Path.cwd()

    # Write output
    output_file = output_dir / "corpus_analysis_report_simple.md"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(report)

    # Print summary
    print("✓ Analysis complete!")
    print(f"  Report: {output_file}")
    print(f"  Files analyzed: {results['total_files']}")
    print(f"  Terms discovered: {len(results['discovered_terms'])}")

    if results["discovered_terms"]:
        print(f"  Coverage: {results['coverage']:.0f}% of files")
        print("\n  Top terms found:")
        for term, count, _ in results["discovered_terms"][:5]:
            pct = count / results["total_files"] * 100
            print(f"    - {term}: {count} files ({pct:.0f}%)")


if __name__ == "__main__":
    main()
