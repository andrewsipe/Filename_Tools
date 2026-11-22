# Changelog

## [Unreleased]

### Changed
- Install and update instructions use `uv tool install` / `uv tool upgrade` (pipx is no longer mentioned).
- `filenametools -h` and the README group the commands by job, in the order a run usually goes (look first, clean up in one go, fix the words, fix hyphens and add Regular, put the terms in order), with a suggested order and a note that everything outside the first group renames files. Command names are unchanged. A few descriptions are in plainer words.

### Fixed
- `filenametools patterns -h` and `filenametools corpus -h` ran the analysis on a path called `-h`. They now print their usage.

## [0.3.0] - 2026-11-21

FontCore 0.4.0 vendored (see `FontCore/VENDOR.md`).

### Added
- `filenametools check`: lists filenames that need a look and changes nothing. Findings: `no-hyphen`, `extra-hyphens`, `spaces`, `weight-in-family`, `cut-short` (suggests the full word from the font's own name records, or marks a dictionary guess as possible), `lowercase-style`, `copy-suffix`, `name-mismatch` (filename style vs the font's name records) and `name-records-odd` (the records hold no style word). On a 1,539-font library sample only 3 fonts are flagged; on a 2,532-font intake copy, 231.

### Changed
- One vocabulary: the slope, width, hyphenation and optical-size lists in the reorder, hyphenation and style-expander commands now come from FontCore, so a word added there reaches every command. `reorder-slopes` now treats `Cursive` and `Kursiv` as slopes (`Probe-CursiveBold` -> `Probe-BoldCursive`). Nothing else changes: every other list holds the same words as before.
- `regular` reads its slope and width words from FontCore too: `Slanted`, `Backslant`, `Backslanted`, `Inclined`, `Kursiv` count as slopes (`Probe-Slanted` -> `Probe-RegularSlanted`) and `Compact`, `Tight` as widths (`Probe-Compact` -> `Probe-CompactRegular`). Its weight list is unchanged: it matches by substring, so the longer FontCore list would catch numbers inside family names.
- `-F` / `--find WORD` (repeatable, or comma-separated) on `reorder-optical` and `reorder-replace`. On `reorder-optical` it looks only for the named words instead of the built-in optical sizes (`--add-optical` still adds to them). On `reorder-replace` it names a whole word to find and move to just after the hyphen, so the command no longer needs `--prehyphen` or `--posthyphen` for that.
- `reorder-replace` closes the gap a moved word leaves: `Probe Foo-Bold` with `-F Foo` is now `Probe-FooBold`, not `Probe -FooBold`. Only names where a word moved are touched.
- `reorder-optical` keeps its filename-only extras (Big, Fine, Head, Large, Small, Tall, Titling) on top of FontCore's list.

## [0.2.2] - 2026-11-21

### Changed
- Vendored FontCore refreshed to v0.3.1 (5 modules). Console wording and counts change ("1 file" / "3 files", `Finished`).
  The style dictionaries the commands read are unchanged (new helpers were added; no word list changed), and the 26 tests pass as before.

## [0.2.1] - 2026-11-20

### Fixed
- `filenametools corpus` no longer raises `ZeroDivisionError` when it finds no terms (for example a folder of
  single-word filenames); the validation summary shows `n/a` percentages.

## [0.2.0] - 2026-11-20

First installable release.

### Changed
- Packaged as `filenametools`, one command: `filenametools <target> ...` (16 targets, one per original script).
  The scripts moved into the `filenametools/` package and keep their names and options.
- FontCore is vendored in `FontCore/` (5 modules, FontCore v0.2.1) instead of a gitignored symlink.
- Python 3.10+ (the code already used 3.10 syntax).

### Fixed
- The `word2number>=1.3` requirement could not be installed (the newest release is 1.1); now `>=1.1`.
- Saves in `compound` and `styles` keep the font's `head.modified` (fonts opened with `recalcTimestamp=False`).
