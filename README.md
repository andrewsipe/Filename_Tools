# Filename Tools

**Version 0.3.0**

Clean, normalize and reorder the terms in font **filenames** so a collection follows one naming pattern. Each tool does one job; `filenametools cleaner` bundles the common ones.

Typical pipeline: **filenametools** → name-table fixes ([FontNameID](https://github.com/andrewsipe/FontNameID)) → rename from metadata ([FileRenamer](https://github.com/andrewsipe/FileRenamer)).

## Install

Preferred: [uv](https://docs.astral.sh/uv/):

```bash
uv tool install "git+https://github.com/andrewsipe/Filename_Tools.git"
# later: uv tool upgrade filenametools
```

From a clone: `uv tool install --editable .` (edits take effect immediately).

Requires Python 3.10+. FontCore is vendored in `FontCore/` (see `FontCore/VENDOR.md`), so there is nothing else to install.

## One command

`filenametools <target> [options] PATH ...` — each target is one of the original single-purpose scripts, options unchanged. `filenametools <target> -h` lists them.

**Look first (these change nothing)**

| Target | Was | What it does |
|---|---|---|
| `check` | `FNT_Check.py` | List filenames that need a look (no hyphen, extra hyphens, stray spaces, cut-short words like `Cond`, filename vs name records) |
| `patterns` | `FNT_PatternAnalyzer.py` | Find the naming patterns in a text file that lists one filename per line |
| `corpus` | `FNT_CorpusPatternAnalyzer.py` | Report which words and word pairs the filenames use |

**Do the usual clean-up in one go**

| Target | Was | What it does |
|---|---|---|
| `cleaner` | `FontFiles_Cleaner.py` | The usual clean-up in one pass, with light to heavy workflows |

**Fix the words**

| Target | Was | What it does |
|---|---|---|
| `normalize` | `FNT_Normalizer.py` | Make a term's spelling and case consistent |
| `compound` | `FNT_CompoundWordNormalizer.py` | Put compound words in the house form (`Semi Bold` → `Semibold`, `Semi Condensed` → `SemiCondensed`, `XX Condensed` → `XXCondensed`) |
| `abbreviations` | `FNT_AbbreviationsExpander.py` | Expand abbreviations (`Reg` → `Regular`) |
| `styles` | `FNT_StyleNameExpander.py` | Expand style-name abbreviations |
| `numbers` | `FNT_NumberWordConverter.py` | Number words to digits (`One` → `1`) |
| `dedupe-words` | `FNT_WordDeduplicator.py` | Remove duplicate words |
| `capitalize-after-hyphen` | `FNT_CapitalizeAfterHyphen.py` | Capitalize words after hyphens |

**Fix hyphens and add Regular**

| Target | Was | What it does |
|---|---|---|
| `hyphenation` | `FNT_HyphenationConsistency.py` | Put hyphens in a consistent place |
| `regular` | `FNT_RegularInserterEnhanced.py` | Insert a missing `Regular` |

**Put the terms in order**

| Target | Was | What it does |
|---|---|---|
| `reorder` | (the four below) | `reorder widths`, `reorder slopes`, `reorder optical`, `reorder replace`: the same as the old names, which still work |
| `reorder-widths` | `FNT_ReorderWidths.py` | Reorder width terms |
| `reorder-slopes` | `FNT_ReorderSlopes.py` | Reorder slope terms |
| `reorder-optical` | `FNT_ReorderOpticalSizes.py` | Reorder optical-size terms |
| `reorder-replace` | `FNT_ReorderFind-n-Replace.py` | Move words you name to just after the hyphen (`-F WORD`) |

Suggested order: `check`, then `cleaner` (or the single commands you need), then `regular`, then the reorder commands. Everything outside the first group renames files on disk, so preview each one with `-n` first. The ones used most are `normalize`, `reorder-widths`, `reorder-slopes` and `capitalize-after-hyphen`.

## Quick start

```bash
filenametools check fonts/ -r               # list names that need a look (renames nothing)
filenametools cleaner fonts/ -r -n          # preview the unified cleaner, recursive
filenametools reorder-widths fonts/ -r -n
filenametools normalize fonts/ -r
```

Most targets take `-r/--recursive`, `-n/--dry-run`, `-v/--verbose` and `--no-preview`. Always run a dry run first: these tools rename files on disk.

The scripts still live in the `filenametools/` package (`python -m filenametools.FNT_Normalizer ...`).

## Notes

- `word2number` is required for the `numbers` target.
- The commands are grouped by job in `filenametools -h` and above. The names are unchanged. Overlaps that remain (the three `reorder-*` commands, `abbreviations` vs `styles`, `compound` vs the house `Semibold` style) are listed in `PRODUCT_REFINEMENT_NOTES.md`.

## Development

```bash
uv venv && uv pip install -e ".[test]" && pytest
```

Refresh the vendored FontCore from inside a FontCore checkout: `python tools/vendor_fontcore.py sync ../Filename_Tools`.
