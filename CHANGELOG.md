# Changelog

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
