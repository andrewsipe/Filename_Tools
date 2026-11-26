# Changelog

## [0.8.0] - 2026-11-26

FontCore 0.11.0 vendored (see `FontCore/VENDOR.md`).

### Changed
- `check` follows the third variable form: `Probe-WidthVariableItalic`, `AugureStereoLight-WidthVariable`, `ParchWeight-CustomVariableUpright` and `ProbeAlt-Variable` are house form; `Probe-VariableWidth`, `Probe-VariableBlack` and `Probe-VariableAlt` are flagged with the new name.

## [0.7.2] - 2026-11-26

FontCore 0.10.2 vendored: a locked `Regular` weight word is kept (`AugureStereo-VariableRegular` is no longer mistaken for a free weight), and is dropped only when the font has a free weight axis.

## [0.7.1] - 2026-11-26

FontCore 0.10.1 vendored: a design-variant word after Variable (Alt, Cut, Swash) moves into the family part (`Probe-VariableAlt` becomes `ProbeAlt-Variable`); Rotalic, Tiltalic and Corsiva are read as slope words.

## [0.7.0] - 2026-11-26

FontCore 0.10.0 vendored (see `FontCore/VENDOR.md`).

### Changed
- `check` follows the refined variable rule: `Probe-VariableUpright`, `-VariableWidthItalic`, `-VariableCustom` and `{Family}Collection-Variable` are house form; the old `Slope`, axis-count and Roman forms are flagged with the new name.

## [0.6.0] - 2026-11-25

FontCore 0.9.1 vendored (`core_variable_filename_parser` added; see `FontCore/VENDOR.md`).

### Changed
- Every renaming command and `cleaner` now leave variable-font filenames alone. They moved a locked width behind the hyphen (`ProbeDisplayCondensed-Variable` became `ProbeDisplay-Condensed-Variable`), which undid the variable form. Variable names are set by `fontrenamer`, which reads the axes from the font.
- `check` judges a variable name by the variable house form `{Family}{Optical}{Width}-Variable{Axes}{Weight}{Slope}`: it flags the old forms (`Probe-CondensedVariable`, `Probe-VariableRegular`, `Probe-VF`, `Probe-Var`) with the suggested name, and a weight before Variable.

### Fixed
- `compound` spelled the Slanted marker `VaraibleSlanted`.

## [0.5.0] - 2026-11-25

FontCore 0.8.2 vendored (see `FontCore/VENDOR.md`).

### Changed
- Every renaming command ends with one sentence ("Updated 3, unchanged 1, no problems."; "Would update ..." in a preview), and the chip turns to a warning when a file had a problem. `cleaner` now uses it too, in place of its own five-line summary.
- Files that failed are no longer counted as unchanged.

## [0.4.1] - 2026-11-25

FontCore 0.7.0 vendored (see `FontCore/VENDOR.md`).

### Changed
- File collection follows one set of rules: macOS junk (`._*`, `__MACOSX`) and hidden files and folders are skipped, each real file is listed once (a symlink and its target are one file), and a path that does not exist gets a warning. Binary formats are the default; this tool asks for TTX where it handles it. The commands still include `.ttx` filenames.

## [0.4.0] - 2026-11-22

FontCore 0.6.0 vendored (see `FontCore/VENDOR.md`).

**Behavior change:** `compound` writes the house form (Semibold, SemiCondensed, XXCondensed); `reorder` is one command for the four reorder commands (old names still work).

### Changed
- Install and update instructions use `uv tool install` / `uv tool upgrade` (pipx is no longer mentioned).
- `filenametools reorder widths|slopes|optical|replace` is one command for the four reorder commands. The old names (`reorder-widths`, ...) still work and run the same code.
- `compound` writes the house form: `Semi Bold` and `SemiBold` become `Semibold`, `Semi Condensed` becomes `SemiCondensed`, `XX Condensed` becomes `XXCondensed` (before, an X run came out as `Xx`), `Semi Mono` becomes `SemiMono`, `Small Caps` becomes `Smallcaps`. The family part is left alone. Its description no longer says `SemiBold -> Semi Bold`.
- Help in plain words for every command (`include subfolders`, `preview only; change nothing`, `skip the preview and start right away`, ...), with a help line for each `cleaner --no-*` flag and for the cleanup flags.
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
