#!/usr/bin/env python3
"""
Font Filename Pattern Analyzer - Sequence Analysis Version

Discovers natural term patterns, frequencies, and positional preferences
in font filename conventions through statistical analysis.

Now includes comprehensive sequence pattern tracking to understand
term ordering and relationships.

Usage:
    python FontFilename_PatternAnalyzer.py [filename_list.txt]
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Set

import FontCore.core_console_styles as cs


class FontPatternAnalyzer:
    """Analyzes font filename patterns to discover natural term groupings."""

    def __init__(self):
        self.total_fonts = 0

        # Raw token tracking (hyphen-separated)
        self.raw_token_frequencies = Counter()
        self.raw_token_positions = defaultdict(list)

        # Atomic token tracking (camelCase-separated)
        self.atomic_term_frequencies = Counter()
        self.atomic_term_positions = defaultdict(list)

        # Relationship tracking
        self.term_cooccurrence = defaultdict(lambda: defaultdict(int))
        self.term_sequences = defaultdict(lambda: defaultdict(int))  # What follows what

        # NEW: Sequence pattern tracking
        self.full_sequences = Counter()  # Complete sequences
        self.sequence_lengths = Counter()  # Distribution of sequence lengths

        # NEW: Term-specific deep tracking
        self.term_details = defaultdict(
            lambda: {
                "predecessors": Counter(),
                "successors": Counter(),
                "co_occurs_with": Counter(),
                "never_with": set(),  # Will be computed later
                "positions": [],
                "as_first": 0,
                "as_last": 0,
                "as_only": 0,
            }
        )

        # Compound tracking
        self.camel_compounds = Counter()  # e.g., "BoldItalic"
        self.hyphen_variants = Counter()  # e.g., "Bold-Italic"

        self.edge_cases = []

        # Patterns
        self.hyphen_split_pattern = re.compile(r"^([^-]+)-(.+)$")
        self.camel_case_pattern = re.compile(r"([A-Z][a-z]*)")

    def decompose_compound(self, token: str) -> List[str]:
        """
        Decompose a camelCase token into atomic parts.
        Example: "BoldItalic" → ["Bold", "Italic"]
        """
        parts = self.camel_case_pattern.findall(token)
        parts = [p for p in parts if len(p) > 1 or p.isupper()]
        return parts if parts else [token]

    def tokenize_style(self, style_text: str) -> List[str]:
        """
        Split on hyphens only.
        Example: "SemiCondensed-BoldItalic" -> ["SemiCondensed", "BoldItalic"]
        """
        if not style_text:
            return []

        tokens = style_text.split("-")
        return [token.strip() for token in tokens if token.strip()]

    def analyze_filename(self, filename: str) -> Dict:
        """Extract family and style tokens from filename."""
        result = {
            "family": "",
            "raw_tokens": [],
            "atomic_tokens": [],
            "is_valid": True,
            "issues": [],
        }

        # Split on first hyphen
        match = self.hyphen_split_pattern.match(filename)
        if not match:
            result["is_valid"] = False
            result["issues"].append("No hyphen separator")
            return result

        family, style = match.groups()
        result["family"] = family

        # Check PascalCase
        if not re.match(r"^[A-Z][a-zA-Z0-9]*$", family):
            result["issues"].append(f"Non-PascalCase family: {family}")

        # Tokenize style portion (hyphen-separated)
        raw_tokens = self.tokenize_style(style)
        result["raw_tokens"] = raw_tokens

        if not raw_tokens:
            result["is_valid"] = False
            result["issues"].append("No style tokens")
            return result

        # Decompose each token into atomic parts
        atomic_tokens = []
        for token in raw_tokens:
            parts = self.decompose_compound(token)
            atomic_tokens.extend(parts)

        result["atomic_tokens"] = atomic_tokens

        return result

    def process_file(self, file_path: Path) -> None:
        """Process the filename list with progress indication."""
        cs.StatusIndicator("info").add_message(
            "Analyzing font filename patterns"
        ).emit()
        cs.StatusIndicator("info").add_file(str(file_path), filename_only=False).emit()

        console = cs.get_console()
        progress = cs.create_progress_bar(console=console)

        with progress:
            task = progress.add_task("Analyzing filenames...", total=None)

            with open(file_path, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    filename = line.strip()
                    if not filename or filename.startswith("#"):
                        continue

                    analysis = self.analyze_filename(filename)
                    self.total_fonts += 1

                    if not analysis["raw_tokens"]:
                        continue

                    atomic_tokens = analysis["atomic_tokens"]
                    seq_length = len(atomic_tokens)

                    # Track sequence length distribution
                    self.sequence_lengths[seq_length] += 1

                    # Track complete sequence
                    sequence_str = " → ".join(atomic_tokens)
                    self.full_sequences[sequence_str] += 1

                    # Track raw tokens (hyphen-separated)
                    for i, token in enumerate(analysis["raw_tokens"]):
                        self.raw_token_frequencies[token] += 1
                        self.raw_token_positions[token].append(i)

                    # Track atomic tokens
                    for i, token in enumerate(atomic_tokens):
                        self.atomic_term_frequencies[token] += 1
                        self.atomic_term_positions[token].append(i)

                        # Track position in sequence
                        details = self.term_details[token]
                        details["positions"].append(i)

                        if i == 0:
                            details["as_first"] += 1
                        if i == seq_length - 1:
                            details["as_last"] += 1
                        if seq_length == 1:
                            details["as_only"] += 1

                        # Track predecessor (if exists)
                        if i > 0:
                            predecessor = atomic_tokens[i - 1]
                            details["predecessors"][predecessor] += 1

                        # Track successor (if exists)
                        if i < seq_length - 1:
                            successor = atomic_tokens[i + 1]
                            details["successors"][successor] += 1

                        # Track what appears in same sequence
                        for other in atomic_tokens:
                            if other != token:
                                details["co_occurs_with"][other] += 1

                    # Track sequences (what follows what)
                    for i in range(len(atomic_tokens) - 1):
                        current = atomic_tokens[i]
                        next_term = atomic_tokens[i + 1]
                        self.term_sequences[current][next_term] += 1

                    # Track co-occurrence
                    if len(atomic_tokens) > 1:
                        for i, token in enumerate(atomic_tokens):
                            for j, other_token in enumerate(atomic_tokens):
                                if i != j:
                                    self.term_cooccurrence[token][other_token] += 1

                    # Track camelCase compounds vs hyphen variants
                    for raw_token in analysis["raw_tokens"]:
                        parts = self.decompose_compound(raw_token)
                        if len(parts) > 1:
                            self.camel_compounds[raw_token] += 1

                            # Check if hyphenated variant exists
                            hyphen_variant = "-".join(parts)
                            if hyphen_variant in analysis["raw_tokens"]:
                                self.hyphen_variants[hyphen_variant] += 1

                    # Track issues
                    if not analysis["is_valid"] or analysis["issues"]:
                        self.edge_cases.append(
                            {"filename": filename, "issues": analysis["issues"]}
                        )

                    if line_num % 1000 == 0:
                        progress.update(
                            task, description=f"Analyzed {cs.fmt_count(line_num)} files"
                        )

        # Compute mutual exclusivity after all data collected
        self._compute_mutual_exclusivity()

        cs.StatusIndicator("success").add_message(
            f"Analysis complete: {cs.fmt_count(self.total_fonts)} fonts"
        ).emit()

    def _compute_mutual_exclusivity(self):
        """Compute which terms never appear together."""
        all_terms = list(self.atomic_term_frequencies.keys())

        for term in all_terms:
            co_occurs = self.term_details[term]["co_occurs_with"]

            for other_term in all_terms:
                if term == other_term:
                    continue

                # If they never appear together
                if co_occurs[other_term] == 0:
                    self.term_details[term]["never_with"].add(other_term)

    def analyze_sequences(self) -> Dict:
        """Analyze sequence patterns."""

        # Group sequences by pattern
        sequence_patterns = defaultdict(list)

        for sequence, count in self.full_sequences.items():
            terms = sequence.split(" → ")
            seq_length = len(terms)

            # Create pattern signature
            pattern_key = f"{seq_length}-term"
            sequence_patterns[pattern_key].append((sequence, count))

        # Sort each pattern group by frequency
        for key in sequence_patterns:
            sequence_patterns[key].sort(key=lambda x: x[1], reverse=True)

        return {
            "full_sequences": self.full_sequences.most_common(50),
            "sequence_lengths": dict(self.sequence_lengths),
            "sequence_patterns": {k: v[:20] for k, v in sequence_patterns.items()},
        }

    def analyze_term_deep(self, term: str) -> Dict:
        """Deep analysis of a single term."""
        if term not in self.term_details:
            return None

        details = self.term_details[term]
        total_count = self.atomic_term_frequencies[term]

        # Calculate position statistics
        positions = details["positions"]
        pos_distribution = Counter(positions)
        avg_pos = sum(positions) / len(positions) if positions else 0

        # Calculate percentages
        first_pct = (details["as_first"] / total_count * 100) if total_count else 0
        last_pct = (details["as_last"] / total_count * 100) if total_count else 0
        only_pct = (details["as_only"] / total_count * 100) if total_count else 0

        return {
            "term": term,
            "frequency": total_count,
            "percentage": round((total_count / self.total_fonts) * 100, 2),
            "position_stats": {
                "average": round(avg_pos, 2),
                "distribution": dict(pos_distribution),
                "as_first": f"{first_pct:.1f}%",
                "as_last": f"{last_pct:.1f}%",
                "as_only": f"{only_pct:.1f}%",
            },
            "predecessors": dict(details["predecessors"].most_common(10)),
            "successors": dict(details["successors"].most_common(10)),
            "never_appears_with": sorted(list(details["never_with"])[:20]),
            "co_occurs_with": dict(details["co_occurs_with"].most_common(10)),
        }

    def detect_modifiers(self) -> Dict[str, Dict]:
        """
        Detect potential modifiers based on:
        - Always appear in compounds (never standalone)
        - Always at position 0 within their compound
        - Limited vocabulary of what follows them
        """
        modifiers = {}

        for term, count in self.atomic_term_frequencies.items():
            details = self.term_details[term]

            # Check if always at position 0
            pos_0_count = details["as_first"]
            if pos_0_count < count * 0.85:  # Must be at pos 0 at least 85% of time
                continue

            # Check what follows this term
            followers = details["successors"]
            if not followers:
                continue

            # Calculate consistency - how often followed by same small set of terms
            total_follows = sum(followers.values())
            top_5_follows = sum(count for _, count in followers.most_common(5))
            follow_consistency = (
                top_5_follows / total_follows if total_follows > 0 else 0
            )

            # High consistency with small vocabulary suggests modifier
            if follow_consistency > 0.7 and len(followers) < 20:
                modifiers[term] = {
                    "count": count,
                    "position_0_ratio": round(pos_0_count / count, 2),
                    "follows_count": len(followers),
                    "follow_consistency": round(follow_consistency, 2),
                    "top_followers": list(followers.most_common(5)),
                }

        return modifiers

    def detect_mutual_exclusivity(self, min_frequency: int = 10) -> List[Set[str]]:
        """
        Detect groups of terms that rarely/never co-occur.
        These likely belong to the same category.
        """
        # Only consider terms above minimum frequency
        frequent_terms = [
            term
            for term, count in self.atomic_term_frequencies.items()
            if count >= min_frequency
        ]

        if len(frequent_terms) < 3:
            return []

        # Build exclusivity matrix using never_with data
        exclusivity = {}
        for term in frequent_terms:
            never_with = self.term_details[term]["never_with"]
            exclusivity[term] = set(t for t in never_with if t in frequent_terms)

        # Find groups using connected components
        groups = []
        visited = set()

        for term in frequent_terms:
            if term in visited:
                continue

            # Build a group of mutually exclusive terms
            group = {term}
            to_check = [term]

            while to_check:
                current = to_check.pop()
                visited.add(current)

                # Find terms exclusive with current
                for other in exclusivity.get(current, set()):
                    if other not in visited and other in frequent_terms:
                        # Check if mutually exclusive with all in group
                        if all(other in exclusivity.get(g, set()) for g in group):
                            group.add(other)
                            to_check.append(other)

            if len(group) > 2:  # Only report groups with 3+ members
                groups.append(group)

        return groups

    def analyze_patterns(self) -> Dict:
        """Run all pattern detection algorithms."""
        cs.StatusIndicator("info").add_message("Detecting patterns...").emit()

        # Analyze sequences
        sequence_analysis = self.analyze_sequences()

        # Deep analysis of top terms
        top_terms_deep = {}
        for term, _ in self.atomic_term_frequencies.most_common(20):
            top_terms_deep[term] = self.analyze_term_deep(term)

        return {
            "total_fonts": self.total_fonts,
            "unique_atomic_terms": len(self.atomic_term_frequencies),
            "unique_raw_tokens": len(self.raw_token_frequencies),
            "sequence_analysis": sequence_analysis,
            "detected_modifiers": self.detect_modifiers(),
            "mutual_exclusivity_groups": [
                list(group) for group in self.detect_mutual_exclusivity()
            ],
            "top_terms_deep_analysis": top_terms_deep,
            "term_frequencies": {
                term: count
                for term, count in self.atomic_term_frequencies.most_common(50)
            },
        }

    def generate_reports(self, output_dir: Path) -> None:
        """Generate JSON and markdown reports."""
        cs.StatusIndicator("info").add_message("Generating reports").emit()

        stats = self.analyze_patterns()

        # JSON report
        json_path = output_dir / "font_pattern_analysis.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        cs.StatusIndicator("saved").add_file(str(json_path)).emit()

        # Markdown report
        md_path = output_dir / "font_pattern_analysis.md"
        self._generate_markdown_report(md_path, stats)
        cs.StatusIndicator("saved").add_file(str(md_path)).emit()

        self._display_summary(stats)

    def _generate_markdown_report(self, file_path: Path, stats: Dict) -> None:
        """Generate comprehensive pattern analysis report."""
        with open(file_path, "w", encoding="utf-8") as f:
            f.write("# Font Filename Pattern Analysis\n\n")

            # Summary
            f.write("## Summary\n\n")
            f.write(f"- **Total fonts analyzed:** {stats['total_fonts']:,}\n")
            f.write(f"- **Unique atomic terms:** {stats['unique_atomic_terms']:,}\n")
            f.write(f"- **Unique raw tokens:** {stats['unique_raw_tokens']:,}\n\n")

            # Sequence Analysis
            seq_analysis = stats["sequence_analysis"]

            f.write("## Sequence Length Distribution\n\n")
            total_seqs = sum(seq_analysis["sequence_lengths"].values())
            for length, count in sorted(seq_analysis["sequence_lengths"].items()):
                pct = (count / total_seqs) * 100
                f.write(f"- **{length}-term sequences:** {count:,} ({pct:.1f}%)\n")
            f.write("\n")

            f.write("## Most Common Sequences\n\n")
            f.write("| Sequence | Count | % |\n")
            f.write("|----------|-------|---|\n")
            for sequence, count in seq_analysis["full_sequences"][:30]:
                pct = (count / stats["total_fonts"]) * 100
                f.write(f"| {sequence} | {count:,} | {pct:.2f}% |\n")
            f.write("\n")

            # Sequence Patterns by Length
            f.write("## Sequence Patterns by Length\n\n")
            for pattern_type in sorted(seq_analysis["sequence_patterns"].keys()):
                f.write(f"### {pattern_type.title()}\n\n")
                for sequence, count in seq_analysis["sequence_patterns"][pattern_type][
                    :15
                ]:
                    pct = (count / stats["total_fonts"]) * 100
                    f.write(f"- {sequence} ({count:,}, {pct:.1f}%)\n")
                f.write("\n")

            # Deep Term Analysis
            f.write("## Deep Term Analysis (Top 20 Terms)\n\n")
            for term, analysis in stats["top_terms_deep_analysis"].items():
                if not analysis:
                    continue

                f.write(f"### {term}\n\n")
                f.write(
                    f"- **Frequency:** {analysis['frequency']:,} ({analysis['percentage']}%)\n"
                )

                pos_stats = analysis["position_stats"]
                f.write(f"- **Position:** Avg {pos_stats['average']}, ")
                f.write(
                    f"First: {pos_stats['as_first']}, Last: {pos_stats['as_last']}, Only: {pos_stats['as_only']}\n"
                )
                f.write(f"- **Position Distribution:** {pos_stats['distribution']}\n")

                if analysis["predecessors"]:
                    pred_str = ", ".join(
                        f"{k} ({v})"
                        for k, v in list(analysis["predecessors"].items())[:5]
                    )
                    f.write(f"- **Preceded by:** {pred_str}\n")
                else:
                    f.write("- **Preceded by:** [nothing]\n")

                if analysis["successors"]:
                    succ_str = ", ".join(
                        f"{k} ({v})"
                        for k, v in list(analysis["successors"].items())[:5]
                    )
                    f.write(f"- **Followed by:** {succ_str}\n")
                else:
                    f.write("- **Followed by:** [nothing]\n")

                if analysis["never_appears_with"]:
                    never_str = ", ".join(analysis["never_appears_with"][:10])
                    f.write(f"- **Never appears with:** {never_str}\n")

                f.write("\n")

            # Detected Modifiers
            f.write("## Detected Modifiers\n\n")
            if stats["detected_modifiers"]:
                f.write(
                    "| Modifier | Count | Pos 0 % | Follows | Consistency | Top Followers |\n"
                )
                f.write(
                    "|----------|-------|---------|---------|-------------|---------------|\n"
                )

                for term, data in sorted(
                    stats["detected_modifiers"].items(),
                    key=lambda x: x[1]["count"],
                    reverse=True,
                ):
                    followers_str = ", ".join(
                        f"{t} ({c})" for t, c in data["top_followers"][:3]
                    )
                    f.write(
                        f"| {term} | {data['count']:,} | {data['position_0_ratio'] * 100:.0f}% | "
                        f"{data['follows_count']} | {data['follow_consistency'] * 100:.0f}% | "
                        f"{followers_str} |\n"
                    )
            else:
                f.write("*No strong modifier patterns detected*\n")

            f.write("\n")

            # Mutual Exclusivity Groups
            f.write("## Mutual Exclusivity Groups\n\n")
            if stats["mutual_exclusivity_groups"]:
                for i, group in enumerate(stats["mutual_exclusivity_groups"], 1):
                    f.write(f"### Group {i}\n\n")
                    sorted_group = sorted(
                        group,
                        key=lambda x: self.atomic_term_frequencies[x],
                        reverse=True,
                    )
                    for term in sorted_group[:20]:
                        count = self.atomic_term_frequencies[term]
                        f.write(f"- **{term}** ({count:,})\n")
                    f.write("\n")
            else:
                f.write("*No mutual exclusivity groups detected*\n\n")

    def _display_summary(self, stats: Dict) -> None:
        """Display key findings to console."""
        cs.StatusIndicator("success").add_message("Pattern Analysis Complete").emit()

        # Show sequence stats
        seq_analysis = stats["sequence_analysis"]
        cs.StatusIndicator("info").add_message(
            f"Found {len(seq_analysis['full_sequences'])} unique sequences"
        ).emit()

        cs.StatusIndicator("info").add_message("\nTop 5 Most Common Sequences:").emit()
        for sequence, count in seq_analysis["full_sequences"][:5]:
            pct = (count / stats["total_fonts"]) * 100
            cs.StatusIndicator("info").add_message(
                f"  {sequence} ({count:,}, {pct:.1f}%)"
            ).emit()

        # Show detected modifiers
        if stats["detected_modifiers"]:
            cs.StatusIndicator("info").add_message(
                f"\nDetected {len(stats['detected_modifiers'])} potential modifiers"
            ).emit()

        # Show mutual exclusivity groups
        if stats["mutual_exclusivity_groups"]:
            cs.StatusIndicator("info").add_message(
                f"\nDetected {len(stats['mutual_exclusivity_groups'])} mutual exclusivity groups"
            ).emit()


def get_input_file() -> Path:
    """Get filename list file from command line or prompt user."""
    if len(sys.argv) > 1:
        input_path = Path(sys.argv[1]).expanduser().resolve()
    else:
        cs.emit(f"\n{cs.INFO_LABEL} Font Filename Pattern Analyzer")
        cs.emit(
            f"{cs.INFO_LABEL} Discovers natural patterns in font naming conventions\n"
        )

        while True:
            file_input = cs.prompt_input("Enter filename list path: ").strip()

            if not file_input:
                cs.emit(f"{cs.WARNING_LABEL} Please enter a file path")
                continue

            file_input = file_input.strip("\"'")
            input_path = Path(file_input).expanduser().resolve()

            if not input_path.exists():
                cs.emit(
                    f"{cs.ERROR_LABEL} File does not exist: {cs.fmt_file(str(input_path))}"
                )
                continue

            if not input_path.is_file():
                cs.emit(
                    f"{cs.ERROR_LABEL} Path is not a file: {cs.fmt_file(str(input_path))}"
                )
                continue

            break

    return input_path


def main():
    """Main script execution."""
    try:
        input_file = get_input_file()
        analyzer = FontPatternAnalyzer()
        analyzer.process_file(input_file)

        output_dir = input_file.parent
        analyzer.generate_reports(output_dir)

        cs.StatusIndicator("success").add_message("Analysis completed!").emit()

    except KeyboardInterrupt:
        cs.StatusIndicator("info").add_message("Analysis cancelled").emit()
    except Exception as e:
        cs.StatusIndicator("error").add_message(f"Error: {e}").emit()
        sys.exit(1)


if __name__ == "__main__":
    main()
