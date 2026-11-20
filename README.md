# Filename Tools

**Version 0.2.0**

Clean, normalize and reorder the terms in font **filenames** so a collection follows one naming pattern. Each tool does one job; `filenametools cleaner` bundles the common ones.

Typical pipeline: **filenametools** → name-table fixes ([FontNameID](https://github.com/andrewsipe/FontNameID)) → rename from metadata ([FileRenamer](https://github.com/andrewsipe/FileRenamer)).

## Install

Preferred: [uv](https://docs.astral.sh/uv/):

```bash
uv tool install "git+https://github.com/andrewsipe/Filename_Tools.git"
# later: uv tool upgrade filenametools
```

Or [pipx](https://pipx.pypa.io/): `pipx install "git+https://github.com/andrewsipe/Filename_Tools.git"`.
From a clone: `uv tool install --editable .` (edits take effect immediately).

Requires Python 3.10+. FontCore is vendored in `FontCore/` (see `FontCore/VENDOR.md`), so there is nothing else to install.

## One command

`filenametools <target> [options] PATH ...` — each target is one of the original single-purpose scripts, options unchanged. `filenametools <target> -h` lists them.

| Target | Was | What it does |
|---|---|---|
| `cleaner` | `FontFiles_Cleaner.py` | Unified cleaner with light/heavy workflows |
| `normalize` | `FNT_Normalizer.py` | Case-insensitive term normalization |
| `numbers` | `FNT_NumberWordConverter.py` | Number words to digits (`One` → `1`) |
| `compound` | `FNT_CompoundWordNormalizer.py` | Compound words (`SemiBold` → `Semi Bold`) |
| `abbreviations` | `FNT_AbbreviationsExpander.py` | Expand abbreviations (`Reg` → `Regular`) |
| `styles` | `FNT_StyleNameExpander.py` | Expand style-name abbreviations |
| `capitalize-after-hyphen` | `FNT_CapitalizeAfterHyphen.py` | Capitalize words after hyphens |
| `hyphenation` | `FNT_HyphenationConsistency.py` | Consistent hyphenation |
| `dedupe-words` | `FNT_WordDeduplicator.py` | Remove duplicate words |
| `regular` | `FNT_RegularInserterEnhanced.py` | Insert a missing `Regular` |
| `reorder-widths` | `FNT_ReorderWidths.py` | Reorder width terms |
| `reorder-slopes` | `FNT_ReorderSlopes.py` | Reorder slope terms |
| `reorder-optical` | `FNT_ReorderOpticalSizes.py` | Reorder optical-size terms |
| `reorder-replace` | `FNT_ReorderFind-n-Replace.py` | Reorder via find/replace patterns |
| `patterns` | `FNT_PatternAnalyzer.py` | Sequence/pattern stats for a filename list |
| `corpus` | `FNT_CorpusPatternAnalyzer.py` | N-gram / corpus report |

## Quick start

```bash
filenametools cleaner fonts/ -r -n          # preview the unified cleaner, recursive
filenametools reorder-widths fonts/ -r -n
filenametools normalize fonts/ -r
```

Most targets take `-r/--recursive`, `-n/--dry-run`, `-v/--verbose` and `--no-preview`. Always run a dry run first: these tools rename files on disk.

The scripts still live in the `filenametools/` package (`python -m filenametools.FNT_Normalizer ...`).

## Notes

- `word2number` is required for the `numbers` target.
- `corpus` raises `ZeroDivisionError` when it finds no comparable files (known, unchanged).
- A more considered grouping of these single-purpose tools is planned; see `PRODUCT_REFINEMENT_NOTES.md`.

## Development

```bash
uv venv && uv pip install -e ".[test]" && pytest
```

Refresh the vendored FontCore from inside a FontCore checkout: `python tools/vendor_fontcore.py sync ../Filename_Tools`.
