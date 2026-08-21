# Product refinement notes — Filename_Tools

Captured during the 2026-08-21 declutter pass. Use for a later product/release pass. **Not** user-facing docs.

## What was archived (declutter)

| Archived path | Was | Why archived |
|---------------|-----|--------------|
| `_misc/_archive/Filename_Tools/FNT_RegularInserter.py` | Width-focused “insert Regular” tool | Overlapped by `FNT_RegularInserterEnhanced.py`; nothing else imports the base file |

No other base/`*Enhanced*` pairs exist in this package. Other scripts are single-purpose CLIs (keep for now).

## Active tree (after declutter)

| Script | Role |
|--------|------|
| `FontFiles_Cleaner.py` | Unified / workflow cleaner (README “main” tool) |
| `FNT_Normalizer.py` | Term find-and-replace normalize |
| `FNT_NumberWordConverter.py` | Number words ↔ digits |
| `FNT_CompoundWordNormalizer.py` | Compound word spacing/casing |
| `FNT_AbbreviationsExpander.py` | Expand abbreviations |
| `FNT_StyleNameExpander.py` | Style abbreviation expand |
| `FNT_CapitalizeAfterHyphen.py` | Cap after hyphen |
| `FNT_HyphenationConsistency.py` | Hyphen consistency |
| `FNT_WordDeduplicator.py` | Duplicate word removal |
| `FNT_RegularInserterEnhanced.py` | Insert missing Regular (canonical) |
| `FNT_ReorderWidths.py` / `FNT_ReorderSlopes.py` / `FNT_ReorderOpticalSizes.py` | Term reorder |
| `FNT_ReorderFind-n-Replace.py` | Reorder via find/replace patterns |
| `FNT_PatternAnalyzer.py` | Sequence/pattern stats on a filename list |
| `FNT_CorpusPatternAnalyzer.py` | N-gram / corpus report (`corpus_analysis_report.md`) |

`FontFiles_Cleaner.py` does **not** subprocess the other `FNT_*` scripts (standalone tools, not a plugin host).

## Behavior to revisit — RegularInserter base vs Enhanced

Enhanced is **not** a strict behavioral superset. Declutter chose Enhanced as the daily driver; product pass should merge or document both models.

### Base (archived) — width-only Regular

- **Scope:** Insert `Regular` after a **leading width** when no weight follows (e.g. `…-Condensed` → `…-CondensedRegular`, `…-CondensedItalic` → `…-CondensedRegularItalic`).
- **Does not** turn bare `Family.otf` into `Family-Regular.otf`.
- **Width dictionary:** `BASE_WIDTH_TERMS` including **Compact**, **Tight**, plus generated `Semi|Extra|Ultra|Super` + base, plus `X`…`XXXXXXX` + base.
- **Slope list:** broader (`Slanted`, `Inclined`, `Backslant*`, `Retalic`, `Smallcaps`, …).
- **CLI:** `--add-slope`, `--add-width`, `--conflict` (no `--add-weight` / `--show-reason`).

### Enhanced (active) — conservative + bare Regular

- **Scope:** Also inserts Regular for **empty style** (`Font.otf` → `Font-Regular.otf`) and width/width+slope cases; skips “ambiguous” optical-ish names (Display / Text / Poster called out in docstring).
- **Width dictionary:** small fixed set (`Condensed`, `Compressed`, `Narrow`, `Extended`, `Expanded`, `Expand`, `Wide`). Comment says modifiers are “handled automatically,” but matching is **prefix against that fixed set** — e.g. **`SemiCondensed` is not recognized as a width** the way the base generator would.
- **Missing vs base widths:** Compact, Tight; full Semi/Extra/Ultra/Super and X-prefix matrices.
- **Weight detection:** substring match on core weights (so `SemiBold` skips via `Bold`).
- **CLI extras:** `--add-weight`, `--show-reason`.

### Product decisions later

1. One Regular-inserter with base’s width matrix **or** Enhanced’s minimal list + bare-family behavior.
2. Whether Compact/Tight / Semi\* widths matter enough to port into Enhanced.
3. Rename drop `Enhanced`; wire console script; align README (still mentioned base name historically).
4. Whether Pattern vs Corpus analyzers stay in the same product as the mutators, or move to a “diagnostics” bundle.
5. `FontFiles_Cleaner` vs discrete `FNT_*` — orchestrate, replace, or keep as parallel entry points.

## Safe non-archive notes

- No metrics-checkpoint junk found in this folder.
- `raw_github_urls.txt` is PushCore noise (same as other packages).
- README omits some active scripts (e.g. `FNT_ReorderSlopes.py`); refresh on product pass.
